import torch
import configargparse
import yaml
import os
import lz4.frame
import zlib
from datetime import datetime

from dataset import build_dataset
from model import build_model
from model.serialization import build_serializer, get_rules_from_args, get_boardsizes_from_args
from utils.training_utils import build_data_loader
from utils.file_utils import find_latest_ckpt, load_torch_ckpt
from utils.misc_utils import set_performance_level, deep_update_dict


def parse_args_and_init():
    parser = configargparse.ArgParser(
        description="Export", config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    parser.add("-c", "--config", is_config_file=True, help="Config file path")
    parser.add("-p", "--checkpoint", help="Model checkpoint file to test")
    parser.add("-o", "--output", help="Output filename")
    parser.add("-r", "--rundir", help="Run directory (specify this if checkpoint is not specified)")
    parser.add("--export_type", required=True, help="Export type")
    parser.add("--export_args", type=yaml.safe_load, default={}, help="Extra export/serialization arguments")
    parser.add("--model_type", required=True, help="Model type")
    parser.add("--model_args", type=yaml.safe_load, default={}, help="Extra model arguments")
    parser.add("--export_model_args", type=yaml.safe_load, default={}, help="Override model args for export")
    parser.add("-d", "--datas", nargs="+", help="Test dataset file or directory paths")
    parser.add(
        "--train_datas",
        nargs="+",
        help="Training dataset file or directory paths (Can be used when data is not specified)",
    )
    parser.add("--dataset_type", help="Dataset type")
    parser.add("--dataset_args", type=yaml.safe_load, default={}, help="Extra dataset arguments")
    parser.add("--dataloader_args", type=yaml.safe_load, default={}, help="Extra dataloader arguments")
    parser.add("--use_cpu", action="store_true", help="Use cpu only")

    args, _ = parser.parse_known_args()  # parse args
    parser.print_values()  # print out values
    print("-" * 60)

    return args


def _get_default_output_filename(checkpoint, ext):
    checkpoint_dir, checkpoint_filename = os.path.split(checkpoint)
    checkpoint_name, checkpoint_ext = os.path.splitext(checkpoint_filename)
    return os.path.join(checkpoint_dir, f"{checkpoint_name}_export.{ext}")


def _get_sample_data(
    datas, dataset_type, dataset_args, dataloader_args, batch_size=1, train_datas=None, **kwargs
):
    # load dataset
    if datas is None and train_datas is not None:
        datas = train_datas
    dataset = build_dataset(dataset_type, datas, shuffle=False, **dataset_args)
    loader = build_data_loader(
        dataset, batch_size=batch_size, num_workers=1, shuffle=False, **dataloader_args
    )

    # get data for tracing
    data = next(iter(loader))
    return data


def _get_git_revision_short_hash(fallback: str = "(unknown)") -> str:
    try:
        import subprocess

        output = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        return output.decode("ascii").strip()
    except:
        return fallback


class ModelIOv1(torch.nn.Module):
    def __init__(self, warpped_model, apply_policy_softmax=False):
        super().__init__()
        self.warpped_model = warpped_model
        self.apply_policy_softmax = apply_policy_softmax
        if apply_policy_softmax:
            print("Warning: Take extra care at the engine side when policy is softmaxed in graph.")

    def get_io_version(self) -> int:
        return 1

    def get_input_names(self):
        return ["board_input", "global_input"]

    def get_output_names(self):
        return ["value", "policy"]

    def get_dynamic_axes(self):
        return {
            "board_input": {0: "batch_size", 2: "board_height", 3: "board_width"},
            "global_input": {0: "batch_size"},
            "value": {0: "batch_size"},
            "policy": {0: "batch_size", 1: "board_height", 2: "board_width"},
        }

    def get_model_inputs(self, data: dict):
        boardInputNCHW = data["board_input"]
        globalInputNC = data["stm_input"]
        assert boardInputNCHW.ndim == 4, f"boardInputNCHW.shape={boardInputNCHW.shape}"
        assert globalInputNC.ndim == 2, f"globalInputNC.shape={globalInputNC.shape}"
        assert boardInputNCHW.shape[0] == globalInputNC.shape[0]
        return boardInputNCHW, globalInputNC

    def get_data_from_model_inputs(self, boardInputNCHW: torch.Tensor, globalInputNC: torch.Tensor):
        return {
            "board_input": boardInputNCHW,
            "stm_input": globalInputNC,
        }

    def get_model_outputs(self, *retvals):
        value, policy, *extra_outputs = retvals
        assert value.ndim == 2 and value.shape[1] == 3, f"value.shape={value.shape}"
        assert policy.ndim == 3, f"policy.shape={policy.shape}"
        assert value.shape[0] == policy.shape[0]
        if self.apply_policy_softmax:
            policy_flat = torch.flatten(policy, start_dim=1)
            policy_flat = torch.nn.functional.softmax(policy_flat, dim=1)
            policy = policy_flat.reshape_as(policy)
        return value, policy

    def forward(self, boardInputNCHW: torch.Tensor, globalInputNC: torch.Tensor):
        input_data = self.get_data_from_model_inputs(boardInputNCHW, globalInputNC)
        retvals = self.warpped_model(input_data)
        return self.get_model_outputs(*retvals)


def _warp_model_io(model, model_io_version=1, apply_policy_softmax=False, **kwargs):
    if model_io_version == 1:
        return ModelIOv1(model, apply_policy_softmax=apply_policy_softmax)
    else:
        raise ValueError(f"Unsupported model IO version {model_io_version}")


def export_torch_jit(output, model, **kwargs):
    # Warp the model with defined input/output interface
    model = _warp_model_io(model, **kwargs)

    # Use the example inputs to trace the model with the given IO
    sampled_data = _get_sample_data(**kwargs)
    example_inputs = model.get_model_inputs(sampled_data)
    scripted_model = torch.jit.trace(model.forward, example_inputs, strict=True)

    # Save the traced model
    torch.jit.save(scripted_model, output)
    print(f"Scripted model has been written to {output}")


def make_onnx_model_version(io_version: int, rules: list[str], boardsizes: list[int]) -> int:
    assert 0 < io_version <= 0xFFFF, f"Invalid IO version {io_version}, must be in [1, 65535]"

    rule_mask = 0
    if "freestyle" in rules:
        rule_mask |= 1
    if "standard" in rules:
        rule_mask |= 2
    if "renju" in rules:
        rule_mask |= 4

    boardsize_mask = 0
    for board_size in boardsizes:
        assert 1 <= board_size <= 32
        boardsize_mask |= 1 << (board_size - 1)

    return (io_version << 48) | (rule_mask << 32) | boardsize_mask


def parse_onnx_model_version(model_version: int) -> tuple[int, list[str], list[int]]:
    io_version = (model_version >> 48) & 0xFFFF
    rule_mask = (model_version >> 32) & 0xFFFF
    boardsize_mask = model_version & 0xFFFFFFFF

    rules = []
    if rule_mask & 1:
        rules.append("freestyle")
    if rule_mask & 2:
        rules.append("standard")
    if rule_mask & 4:
        rules.append("renju")

    boardsizes = []
    for i in range(32):
        if boardsize_mask & (1 << i):
            boardsizes.append(i + 1)

    return io_version, rules, boardsizes


def export_onnx(output, model, export_args, **kwargs):
    import onnx

    # Warp the model with defined input/output interface
    model = _warp_model_io(model, **kwargs)

    # Output model to ONNX format
    model.eval()
    sampled_data = _get_sample_data(**kwargs)
    args = model.get_model_inputs(sampled_data)
    torch.onnx.export(
        model,
        args,
        output,
        export_params=True,
        do_constant_folding=True,
        input_names=model.get_input_names(),
        output_names=model.get_output_names(),
        dynamic_axes=model.get_dynamic_axes(),
        dynamo=export_args.get("onnx_use_dynamo", False),
    )

    # Run OnnxSlim if available
    try:
        import onnxslim

        print("Running onnxslim to optimize the model...")
        onnxslim.slim(output, output)
    except:
        pass

    # Add metadata to the exported ONNX model
    io_version = model.get_io_version()
    supported_rules = get_rules_from_args(export_args)
    supported_boardsizes = get_boardsizes_from_args(export_args)
    onnx_model = onnx.load(output)
    onnx_model.model_version = make_onnx_model_version(io_version, supported_rules, supported_boardsizes)
    onnx_model.producer_name = "https://github.com/dhbloo/pytorch-nnue-trainer"
    onnx_model.producer_version = _get_git_revision_short_hash()
    onnx.save(onnx_model, output)
    print(
        f"Onnx metadata: "
        f"\nmodel_version={hex(onnx_model.model_version)} "
        f"\n(io_ver={io_version}, rules={supported_rules}, boardsizes={supported_boardsizes}) "
        f"\nproducer_name={onnx_model.producer_name} "
        f"\nproducer_version={onnx_model.producer_version} "
        f"\nir_version={onnx_model.ir_version} "
    )

    print(f"Onnx model has been exported to {output}.")


def export_serialization(
    output, output_type, model_type, model, export_args, use_cpu, no_header=False, **kwargs
):
    serializer = build_serializer(model_type, **export_args)
    device = torch.device("cuda" if not use_cpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    def _open_output_file(output_path):
        file_mode = "w" + ("b" if serializer.is_binary else "t")
        if output_type == "raw":
            return open(output_path, file_mode)
        elif output_type == "lz4":
            return lz4.frame.open(
                output_path,
                file_mode,
                compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC,
                content_checksum=True,
            )
        else:
            raise ValueError(f"Unsupported serialization output type {output_type}")

    with _open_output_file(output) as f:
        # serialize header for binary weight format
        if serializer.needs_header and not no_header:
            MAGIC = zlib.crc32(b"gomoku network weight version 1")  # 0xacd8cc6a
            arch_hash = serializer.arch_hash(model) & 0xFFFFFFFF
            rule_mask = serializer.rule_mask(model) & 0xFFFFFFFF
            boardsize_mask = serializer.boardsize_mask(model) & 0xFFFFFFFF
            description = serializer.description(model)
            timestamp_str = datetime.now().strftime("%c")
            commit_hash = _get_git_revision_short_hash()
            description += (
                f"Weight exported by pytorch-nnue-trainer (commit hash {commit_hash}) at {timestamp_str}."
            )
            encoded_description = description.encode("utf-8")

            f.write(MAGIC.to_bytes(4, byteorder="little", signed=False))
            f.write(arch_hash.to_bytes(4, byteorder="little", signed=False))
            f.write(rule_mask.to_bytes(4, byteorder="little", signed=False))
            f.write(boardsize_mask.to_bytes(4, byteorder="little", signed=False))
            f.write(len(encoded_description).to_bytes(4, byteorder="little", signed=False))
            f.write(encoded_description)

            print(
                f"Write serializer header: "
                f"\nmagic = {hex(MAGIC)}"
                f"\narch_hash = {hex(arch_hash)}"
                f"\nrule_mask = {hex(rule_mask)}"
                f"\nboardsize_mask = {hex(boardsize_mask)}"
                f'\ndescription = "{description}"'
            )

        with torch.no_grad():
            serializer.serialize(f, model, device)

    type = "binary" if serializer.is_binary else "text"
    print(f"Serialized {type} model has been exported to {output}")


def export(checkpoint, output, rundir, export_type, model_type, model_args, export_model_args, **kwargs):
    set_performance_level(0)
    # construct the model
    model = build_model(model_type, **model_args)
    model_name = model.name
    if export_model_args:
        model_args = deep_update_dict(model_args, export_model_args)
        model = build_model(model_type, **model_args)

    if checkpoint is None:
        if rundir is None:
            raise RuntimeError("Either checkpoint or rundir must be specified.")
        # try to find the latest checkpoint
        checkpoint = find_latest_ckpt(rundir, f"ckpt_{model_name}")
        if checkpoint is None:
            raise RuntimeError(f"No checkpoint found in {rundir} (prefix=ckpt_{model_name})")
    elif not os.path.exists(checkpoint) or not os.path.isfile(checkpoint):
        raise RuntimeError(f"Checkpoint {checkpoint} must be a valid file")

    # load checkpoint
    print(f"Loading model state from {checkpoint}")
    model_state_dict, _, _ = load_torch_ckpt(checkpoint)
    model.load_state_dict(model_state_dict)
    model.eval()

    if export_type == "torch-jit":
        output = output or _get_default_output_filename(checkpoint, "pt")
        export_torch_jit(output, model, **kwargs)
    elif export_type == "onnx":
        output = output or _get_default_output_filename(checkpoint, "onnx")
        export_onnx(output, model, **kwargs)
    elif export_type == "txt":
        output = output or _get_default_output_filename(checkpoint, "txt")
        kwargs["export_args"].setdefault("text_output", True)
        export_serialization(output, "raw", model_type, model, **kwargs)
    elif export_type == "bin":
        output = output or _get_default_output_filename(checkpoint, "bin")
        export_serialization(output, "raw", model_type, model, **kwargs)
    elif export_type == "bin-noheader":
        output = output or _get_default_output_filename(checkpoint, "bin")
        export_serialization(output, "raw", model_type, model, **kwargs, no_header=True)
    elif export_type == "bin-lz4":
        output = output or _get_default_output_filename(checkpoint, "bin.lz4")
        export_serialization(output, "lz4", model_type, model, **kwargs)
    elif export_type == "bin-lz4-noheader":
        output = output or _get_default_output_filename(checkpoint, "bin.lz4")
        export_serialization(output, "lz4", model_type, model, **kwargs, no_header=True)
    else:
        raise ValueError(f"Unsupported export: {export_type}")


if __name__ == "__main__":
    args = parse_args_and_init()
    export(**vars(args))
