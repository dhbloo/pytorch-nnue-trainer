"""Standard ONNX SizeBN lowering and per-board session specialization.

ONNX and ONNX Runtime are optional dependencies, imported only when needed.
"""

import json


_SIZEBN_METADATA = "ntr.sizebn.v1"


def lower_sizebn_onnx(model, board_sizes):
    """Lower export markers to shape selection and affine arithmetic (opset 18)."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    nodes, selectors = [], []
    for marker in model.graph.node:
        if marker.domain != "ntr" or marker.op_type != "SizeBNAffine":
            nodes.append(marker)
            continue
        prefix = marker.output[0] + "/sizebn/"

        def const(name, value):
            name = prefix + name
            model.graph.initializer.append(numpy_helper.from_array(np.asarray(value, np.int64), name))
            return name

        def node(op, inputs, output, **attrs):
            output = prefix + output
            nodes.append(helper.make_node(op, inputs, [output], name=output, **attrs))
            return output

        x, scales, biases, sizes = marker.input
        axes1, axes0 = const("axes1", [1]), const("axes0", [0])
        one = const("one", 1)
        valid_shape = const("valid_shape", [1, -1, 1, 1])
        invalid_shape = const("invalid_shape", [-1, -1, 1, 1])
        hw = node("Shape", [x], "hw", start=2, end=4)
        matches = node("Equal", [sizes, hw], "matches")
        matches = node("Cast", [matches], "matches_i64", to=TensorProto.INT64)
        matched = node("ReduceMin", [matches, axes1], "matched", keepdims=0)
        count = node("ReduceSum", [matched, axes0], "count", keepdims=0)
        valid = node("Equal", [count, one], "valid")
        bucket = node("ArgMax", [matched], "bucket", axis=0, keepdims=0)
        # CUDA Gather does not reliably reject out-of-range indices. Reshape's
        # shape validation rejects two inferred dimensions on both CPU and CUDA.
        affine_shape = node("Where", [valid, valid_shape, invalid_shape], "affine_shape")
        scale = node("Gather", [scales, bucket], "scale", axis=0)
        bias = node("Gather", [biases, bucket], "bias", axis=0)
        scale = node("Reshape", [scale, affine_shape], "scale4")
        bias = node("Reshape", [bias, affine_shape], "bias4")
        scaled = node("Mul", [x, scale], "scaled")
        nodes.append(helper.make_node("Add", [scaled, bias], marker.output, name=prefix + "affine"))
        selectors.append({"shape_output": hw, "activation": x,
                          "sizes": [[s, s] for s in board_sizes]})
    if not selectors:
        raise ValueError("Dynamic sizebn export produced no SizeBNAffine markers")
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    imports = [entry for entry in model.opset_import if entry.domain != "ntr"]
    del model.opset_import[:]
    model.opset_import.extend(imports)
    metadata = {p.key: p.value for p in model.metadata_props}
    metadata[_SIZEBN_METADATA] = json.dumps({"board_sizes": board_sizes, "selectors": selectors})
    helper.set_model_props(model, metadata)
    return model


def specialize_sizebn_onnx(model_path, board_size):
    """Load a dynamic export and freeze its size selection, leaving batch dynamic.

    The source file is unchanged. Run this once per board size, before creating
    an optimized inference session. Keep the exported selector metadata intact;
    externally rewritten/optimized graphs are not supported as source artifacts.
    """
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    if isinstance(board_size, bool) or not isinstance(board_size, int):
        raise ValueError("board_size must be a single integer")
    model = onnx.load(model_path)
    metadata = {p.key: p.value for p in model.metadata_props}
    if _SIZEBN_METADATA not in metadata:
        raise ValueError("Expected a dynamic sizebn ONNX export with selector metadata")
    spec = json.loads(metadata[_SIZEBN_METADATA])
    if "specialized_board_size" in spec:
        raise ValueError("Specialize the original dynamic export, not an already specialized model")
    if board_size not in spec["board_sizes"]:
        raise ValueError(f"Unsupported board size {board_size}; supported: {spec['board_sizes']}")
    board = next((v for v in model.graph.input if v.name == "board_input"), None)
    if board is None or len(board.type.tensor_type.shape.dim) != 4:
        raise ValueError("Expected rank-4 board_input")
    board.type.tensor_type.shape.dim[2].dim_value = board_size
    board.type.tensor_type.shape.dim[3].dim_value = board_size
    # Discard intermediate trace shapes so inference derives each activation's
    # actual H/W from the fixed input, including spatially transforming layers.
    del model.graph.value_info[:]
    for value in model.graph.output:
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param in ("board_height", "board_width"):
                dim.dim_value = board_size
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    values = {v.name: v for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
    selector_outputs = set()
    for selector in spec["selectors"]:
        output, activation = selector["shape_output"], selector["activation"]
        matches = [n for n in model.graph.node if output in n.output]
        if len(matches) != 1 or matches[0].op_type != "Shape" or list(matches[0].input) != [activation]:
            raise ValueError(f"Sizebn selector was rewritten or removed: {output}")
        value = values.get(activation)
        dims = [] if value is None else value.type.tensor_type.shape.dim
        if len(dims) != 4 or not all(d.HasField("dim_value") for d in dims[2:]):
            raise ValueError(f"Cannot infer sizebn activation H/W: {activation}")
        hw = [dims[2].dim_value, dims[3].dim_value]
        if hw not in selector["sizes"]:
            raise ValueError(f"Sizebn activation {activation} has no trained bucket for {hw}")
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(hw, np.int64), output))
        selector_outputs.add(output)
    nodes = [n for n in model.graph.node if not selector_outputs.intersection(n.output)]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    spec["specialized_board_size"] = board_size
    metadata[_SIZEBN_METADATA] = json.dumps(spec)
    helper.set_model_props(model, metadata)
    model.model_version = (model.model_version & ~0xFFFFFFFF) | (1 << (board_size - 1))
    onnx.checker.check_model(model)
    return model


def create_sizebn_session(model_path, board_size, providers=None, sess_options=None):
    """Create an ORT session whose size lookup is constant-folded at load time.

    Cache returned sessions by board size. Batch size and board contents remain
    runtime inputs. Supply the same providers/options used for other sessions;
    graph optimization must be enabled for the lookup to disappear.
    """
    import onnxruntime as ort

    model = specialize_sizebn_onnx(model_path, board_size)
    return ort.InferenceSession(model.SerializeToString(), sess_options=sess_options, providers=providers)
