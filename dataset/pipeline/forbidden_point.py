import numpy as np
from . import BasePipeline, PIPELINES
from dataset.core import FieldSpec


@PIPELINES.register("forbidden_point")
class ForbiddenPointPipeline(BasePipeline):
    pipeline_id = "forbidden_point"
    schema_version = 1
    parallel_stateless = True
    input_fields = (
        FieldSpec("board_input", True, "per_sample", (-2, -1), 0, "stack", ("b", "i", "u"), "plain"),
        FieldSpec("stm_input", True, "per_sample", None, None, "stack", ("i", "f"), "plain"),
    )
    output_fields = (
        FieldSpec("forbidden_point", True, "per_sample", (-2, -1), None, "stack", ("b", "i", "u"), "plain"),
    )
    def __init__(self, fixed_side_input) -> None:
        super().__init__()
        self.fixed_side_input = fixed_side_input

    def signature_state(self):
        return {"fixed_side_input": bool(self.fixed_side_input)}

    def added_output_row_bytes(self, board_size):
        height, width = board_size
        # One int8 output plane plus at most one contiguous int8 board copy.
        return 3 * height * width

    def process(self, data):
        from forbidden_point_cpp import transform_board_to_forbidden_point

        board_input = data["board_input"]  # [2, H, W]
        _, H, W = board_input.shape

        # ensure that black side is at channel 0
        if not self.fixed_side_input:
            stm_is_black = data["stm_input"] < 0
            # swap side if side to move is white
            if not stm_is_black:
                board_input = np.flip(board_input, axis=0)

        # allocate space for forbidden point results
        forbidden_point = np.empty((H, W), dtype=np.int8)  # [H, W]

        # do feature transformation
        transform_board_to_forbidden_point(board_input, forbidden_point)

        # add forbidden point flags to data
        data = dict(data)
        data["forbidden_point"] = forbidden_point
        return data

    def process_batch(self, data, *, sample_keys=None, rng_keys=None):
        from forbidden_point_cpp import transform_board_to_forbidden_point

        boards = data["board_input"]
        height, width = boards.shape[-2:]
        outputs = np.empty((len(boards), height, width), dtype=np.int8)
        for index, board in enumerate(boards):
            if not self.fixed_side_input and not (data["stm_input"][index] < 0):
                board = np.flip(board, axis=0)
            transform_board_to_forbidden_point(board, outputs[index])
        result = dict(data)
        result["forbidden_point"] = outputs
        return result
