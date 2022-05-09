import numpy as np
from . import BasePipeline, PIPELINES


@PIPELINES.register('forbidden_point')
class ForbiddenPointPipeline(BasePipeline):
    def __init__(self, fixed_side_input) -> None:
        super().__init__()
        self.fixed_side_input = fixed_side_input

    def process(self, data):
        from forbidden_point_cpp import transform_board_to_forbidden_point

        board_input = data['board_input']  # [2, H, W]
        _, H, W = board_input.shape

        # ensure that black side is at channel 0
        if not self.fixed_side_input:
            stm_is_black = data['stm_input'] < 0
            # swap side if side to move is white
            if not stm_is_black:
                board_input = np.flip(board_input, axis=0)

        # allocate space for forbidden point results
        forbidden_point = np.empty((H, W), dtype=np.int8)  # [H, W]

        # do feature transformation
        transform_board_to_forbidden_point(board_input, forbidden_point)

        # add forbidden point flags to data
        data['forbidden_point'] = forbidden_point
        return data