#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "board.h"

namespace py = pybind11;

/// Transform a board input numpy array to 4 direction line encoding output numpy array
/// @param board_input Board numpy array of shape [2, H, W]. Note that the first channel
///     should always be black, while the second channel should always be white.
/// @param forbidden_point_output Forbidden point flag numpy array of shape [H, W].
void transform_board_to_forbidden_point(
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> board_input,
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> forbidden_point_output)
{
    auto board = board_input.unchecked<3>();
    auto forbidden_point = forbidden_point_output.mutable_unchecked<2>();
    int H = (int)board.shape(1), W = (int)board.shape(2);

    // Check input legality
    if (board.shape(0) != 2)
        throw std::invalid_argument("board shape incorrect, must be [2,H,W]");
    if (forbidden_point.shape(0) != H || forbidden_point.shape(1) != W)
        throw std::invalid_argument("forbidden_point shape incorrect, must be [H,W]");
    if (H > Board::MaxRealBoardSize || W > Board::MaxRealBoardSize)
        throw std::invalid_argument("board size must be less or equal to " + std::to_string(Board::MaxRealBoardSize));

    Board fp_board(W, H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
        {
            bool is_black = board(0, y, x);
            bool is_white = board(1, y, x);
            Color color = is_black ? BLACK : is_white ? WHITE
                                                      : EMPTY;
            if (color != EMPTY)
            {
                Pos pos = MakePos(x, y);
                fp_board.set(pos, color);
            }
        }

    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            forbidden_point(y, x) = fp_board.isForbidden(MakePos(x, y));
}

using namespace py::literals;

PYBIND11_MODULE(forbidden_point_cpp, m)
{
    m.doc() = "Transform board input to forbidden point flag";
    m.def("transform_board_to_forbidden_point", &transform_board_to_forbidden_point,
          "board_input"_a,
          "forbidden_point_output"_a);
}