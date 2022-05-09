#include <cstdint>
#include <cassert>

#define BOARD_BOUNDARY 5
#define MAX_BOARD_SIZE_BIT 5

enum Color
{
    BLACK,
    WHITE,
    EMPTY,
    WALL,
    COLOR_NB = 2
};

constexpr Color Opponent(Color c) { return Color(c ^ 1); }

enum ForbiddenType
{
    FORBIDDEN_NONE,
    DOUBLE_THREE,
    DOUBLE_FOUR,
    OVERLINE
};

typedef uint16_t Pos;

constexpr Pos MakePos(int x, int y)
{
    x = x + BOARD_BOUNDARY;
    y = y + BOARD_BOUNDARY;
    return (x << MAX_BOARD_SIZE_BIT) + y;
}

constexpr int CoordX(Pos pos)
{
    return (pos >> MAX_BOARD_SIZE_BIT) - BOARD_BOUNDARY;
}

constexpr int CoordY(Pos pos)
{
    return (pos & ((1 << MAX_BOARD_SIZE_BIT) - 1)) - BOARD_BOUNDARY;
}

class Board
{
public:
    static constexpr int MaxBoardSize = 1 << MAX_BOARD_SIZE_BIT;
    static constexpr int MaxBoardCellCount = MaxBoardSize * MaxBoardSize;
    static constexpr int MaxRealBoardSize = MaxBoardSize - 2 * BOARD_BOUNDARY;

    Board(int width, int height);

    bool isValid(Pos pos) const { return pos < MaxBoardCellCount; }
    bool isEmpty(Pos pos) const { return get(pos) == EMPTY; }
    Color get(Pos pos) const
    {
        assert(isValid(pos));
        return board[pos];
    }
    void set(Pos pos, Color color)
    {
        assert(isValid(pos));
        assert(get(pos) != WALL);
        board[pos] = color;
    }

    ForbiddenType isForbidden(Pos pos);

private:
    Color board[MaxBoardCellCount];
    int boardWidth, boardHeight;

    // renju helpers
    enum OpenFourType
    {
        OF_NONE,
        OF_TRUE /*_OOOO_*/,
        OF_LONG /*O_OOO_O*/,
    };
    bool isFive(Pos pos, Color piece);
    bool isFive(Pos pos, Color piece, int iDir);
    bool isOverline(Pos pos, Color piece);
    bool isFour(Pos pos, Color piece, int iDir);
    OpenFourType isOpenFour(Pos pos, Color piece, int iDir);
    bool isOpenThree(Pos pos, Color piece, int iDir);
    bool isDoubleFour(Pos pos, Color piece);
    bool isDoubleThree(Pos pos, Color piece);
};