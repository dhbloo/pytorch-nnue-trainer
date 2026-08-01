import numpy as np
from . import BasePipeline, PIPELINES
from dataset.core import FieldSpec


@PIPELINES.register("line_encoding")
class LineEncodingPipeline(BasePipeline):
    pipeline_id = "line_encoding"
    schema_version = 1
    input_fields = (
        FieldSpec("board_input", True, "per_sample", (-2, -1), 0, "stack", ("b", "i", "u"), "plain"),
    )
    output_fields = (
        FieldSpec("line_encoding", True, "per_sample", (-2, -1), None, "stack", ("i",), "plain"),
        FieldSpec(
            "line_encoding_total_num", True, "batch_shared", None, None,
            "broadcast", ("i", "u"), "plain"
        ),
    )
    def __init__(self, line_length=11, raw_code=False) -> None:
        from line_encoding_cpp import get_total_num_encoding

        super().__init__()
        if line_length < 1 or line_length % 2 != 1:
            raise ValueError("line_length must be a positive odd number")
        if not raw_code and line_length > 17:
            raise ValueError("compressed line_length must not exceed 17")
        if raw_code and line_length > 15:
            raise ValueError("raw-code line_length must not exceed 15")
        self.line_length = line_length
        self.raw_code = raw_code
        self.line_encoding_total_num = get_total_num_encoding(self.line_length) if not raw_code else 4**line_length

    def signature_state(self):
        return {"line_length": self.line_length, "raw_code": self.raw_code}

    def process(self, data):
        from line_encoding_cpp import transform_board_to_line_encoding

        board_input = data["board_input"]  # [2, H, W]
        _, H, W = board_input.shape

        # allocate space for line encoding results
        line_encoding = np.empty((4, H, W), dtype=np.int32)  # [4, H, W]

        # do feature transformation
        transform_board_to_line_encoding(board_input, line_encoding, self.line_length, raw_code=self.raw_code)

        # add line encoding to data
        data = dict(data)
        data["line_encoding"] = line_encoding
        data["line_encoding_total_num"] = self.line_encoding_total_num
        return data

    def process_batch(self, data, *, sample_keys=None, rng_keys=None):
        """Vectorized batch variant used by batch-yielding datasets."""
        from line_encoding_cpp import transform_boards_to_line_encoding

        board_input = data["board_input"]  # [B, 2, H, W]
        B, _, H, W = board_input.shape
        line_encoding = np.empty((B, 4, H, W), dtype=np.int32)
        transform_boards_to_line_encoding(
            board_input, line_encoding, self.line_length, raw_code=self.raw_code
        )
        data = dict(data)
        data["line_encoding"] = line_encoding
        # Per-sample processing produces a scalar that default_collate turns
        # into [B]. Preserve that contract for models that validate this field.
        data["line_encoding_total_num"] = np.full(
            B, self.line_encoding_total_num, dtype=np.int64
        )
        return data


def get_total_num_encoding(line_length: int) -> int:
    """Get total number of encoding for a line of given length."""
    from line_encoding_cpp import get_total_num_encoding

    return get_total_num_encoding(line_length)


def get_encoding_usage_flags(line_length: int) -> np.ndarray:
    """
    Get encoding usage flags of a encoding map.
    Returns: int8 np.ndarray of shape (total_num_encoding,)
    """
    from line_encoding_cpp import get_total_num_encoding, get_encoding_usage_flag

    total_num_encoding = get_total_num_encoding(line_length)
    usage_flags = np.zeros(total_num_encoding, dtype=np.int8)
    get_encoding_usage_flag(usage_flags, line_length)

    return usage_flags


def transform_lines_to_line_encoding(lines_input: np.ndarray, line_length: int) -> np.ndarray:
    """
    Get line encoding of the given batched lines.
    Args:
        lines_input: Lines int8 array of shape [N, L].
            Elements are in {0,1,2} for empty/self/oppo.
        line_length: Length of line encoding.
    Returns:
        line_encodings: Line encoding int32 array of shape [N, L].
    """
    from line_encoding_cpp import transform_lines_to_line_encoding

    if lines_input.ndim != 2 or lines_input.dtype != np.int8:
        raise ValueError("lines_input must be a two-dimensional int8 array")
    if np.min(lines_input) < 0 or np.max(lines_input) > 2:
        raise ValueError("lines_input values must be in [0, 2]")

    line_encodings_output = np.zeros_like(lines_input, dtype=np.int32)
    transform_lines_to_line_encoding(lines_input, line_encodings_output, line_length)

    return line_encodings_output
