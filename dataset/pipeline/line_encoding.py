import numpy as np
from . import BasePipeline, PIPELINES


@PIPELINES.register("line_encoding")
class LineEncodingPipeline(BasePipeline):
    def __init__(self, line_length=11, raw_code=False) -> None:
        from line_encoding_cpp import get_total_num_encoding

        super().__init__()
        assert line_length % 2 == 1, "Length must be a odd number"
        assert line_length < 20, "Max supported line length is 19"
        assert not raw_code or line_length <= 15, "Max supported line length for raw code is 15"
        self.line_length = line_length
        self.raw_code = raw_code
        self.line_encoding_total_num = get_total_num_encoding(self.line_length) if not raw_code else 4**line_length

    def process(self, data):
        from line_encoding_cpp import transform_board_to_line_encoding

        board_input = data["board_input"]  # [2, H, W]
        _, H, W = board_input.shape

        # allocate space for line encoding results
        line_encoding = np.empty((4, H, W), dtype=np.int32)  # [4, H, W]

        # do feature transformation
        transform_board_to_line_encoding(board_input, line_encoding, self.line_length, raw_code=self.raw_code)

        # add line encoding to data
        data["line_encoding"] = line_encoding
        data["line_encoding_total_num"] = self.line_encoding_total_num
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

    assert lines_input.ndim == 2 and lines_input.dtype == np.int8
    assert np.min(lines_input) >= 0 and np.max(lines_input) <= 2

    line_encodings_output = np.zeros_like(lines_input, dtype=np.int32)
    transform_lines_to_line_encoding(lines_input, line_encodings_output, line_length)

    return line_encodings_output
