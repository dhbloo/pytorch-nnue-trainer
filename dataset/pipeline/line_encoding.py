import numpy as np
from . import BasePipeline, PIPELINES


@PIPELINES.register('line_encoding')
class LineEncodingPipeline(BasePipeline):
    def __init__(self, line_length=11) -> None:
        from line_encoding_cpp import get_total_num_encoding
        
        super().__init__()
        assert line_length % 2 == 1, 'Length must be a odd number'
        assert line_length < 20, 'Max supported line length is 19'
        self.line_length = line_length
        self.line_encoding_total_num = get_total_num_encoding(self.line_length)

    def process(self, data):
        from line_encoding_cpp import transform_board_to_line_encoding

        board_input = data['board_input']  # [2, H, W]
        _, H, W = board_input.shape

        # allocate space for line encoding results
        line_encoding = np.empty((4, H, W), dtype=np.int32)  # [4, H, W]

        # do feature transformation
        transform_board_to_line_encoding(board_input, line_encoding, self.line_length)

        # add line encoding to data
        data['line_encoding'] = line_encoding
        data['line_encoding_total_num'] = self.line_encoding_total_num
        return data