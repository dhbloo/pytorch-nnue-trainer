#include "board.h"

typedef int16_t Direction;
const Direction DIRECTION[4] = {1,
                                Board::MaxBoardSize - 1,
                                Board::MaxBoardSize,
                                Board::MaxBoardSize + 1};

Board::Board(int width, int height) : boardWidth(width), boardHeight(height)
{
    for (Pos pos = 0; pos < MaxBoardCellCount; pos++)
    {
        int x = CoordX(x), y = CoordY(y);
        board[pos] = x >= 0 && x < boardWidth && y >= 0 && y < boardHeight ? EMPTY : WALL;
    }
}

ForbiddenType Board::isForbidden(Pos pos)
{
    if (isDoubleThree(pos, BLACK))
        return DOUBLE_THREE;
    else if (isDoubleFour(pos, BLACK))
        return DOUBLE_FOUR;
    else if (isOverline(pos, BLACK))
        return OVERLINE;
    else
        return FORBIDDEN_NONE;
}

bool Board::isFive(Pos pos, Color piece)
{
    if (board[pos] != EMPTY)
        return false;

    for (int iDir = 0; iDir < 4; iDir++)
    {
        if (isFive(pos, piece, iDir))
            return true;
    }
    return false;
}

bool Board::isFive(Pos pos, Color piece, int iDir)
{
    if (board[pos] != EMPTY)
        return false;

    int i, j;
    int count = 1;
    for (i = 1; i < 6; i++)
    {
        if (board[pos - DIRECTION[iDir] * i] == piece)
            count++;
        else
            break;
    }
    for (j = 1; j < 7 - i; j++)
    {
        if (board[pos + DIRECTION[iDir] * j] == piece)
            count++;
        else
            break;
    }
    return count == 5;
}

bool Board::isOverline(Pos pos, Color piece)
{
    if (board[pos] != EMPTY)
        return false;

    for (Direction dir : DIRECTION)
    {
        int i, j;
        int count = 1;
        for (i = 1; i < 6; i++)
        {
            if (board[pos - dir * i] == piece)
                count++;
            else
                break;
        }
        for (j = 1; j < 7 - i; j++)
        {
            if (board[pos + dir * j] == piece)
                count++;
            else
                break;
        }
        if (count > 5)
            return true;
    }
    return false;
}

bool Board::isFour(Pos pos, Color piece, int iDir)
{
    if (board[pos] != EMPTY)
        return false;
    else if (isFive(pos, piece))
        return false;
    else if (piece == BLACK && isOverline(pos, BLACK))
        return false;
    else if (piece == BLACK || piece == WHITE)
    {
        bool four = false;
        set(pos, piece);

        int i, j;
        for (i = 1; i < 5; i++)
        {
            Pos posi = pos - DIRECTION[iDir] * i;
            if (board[posi] == piece)
                continue;
            else if (board[posi] == EMPTY && isFive(posi, piece, iDir))
                four = true;
            break;
        }
        for (j = 1; !four && j < 6 - i; j++)
        {
            Pos posi = pos + DIRECTION[iDir] * j;
            if (board[posi] == piece)
                continue;
            else if (board[posi] == EMPTY && isFive(posi, piece, iDir))
                four = true;
            break;
        }

        set(pos, EMPTY);
        return four;
    }
    else
        return false;
}

Board::OpenFourType Board::isOpenFour(Pos pos, Color piece, int iDir)
{
    if (board[pos] != EMPTY)
        return OF_NONE;
    else if (isFive(pos, piece))
        return OF_NONE;
    else if (piece == BLACK && isOverline(pos, BLACK))
        return OF_NONE;
    else if (piece == BLACK || piece == WHITE)
    {
        set(pos, piece);

        int i, j;
        int count = 1;
        int five = 0;

        for (i = 1; i < 5; i++)
        {
            Pos posi = pos - DIRECTION[iDir] * i;
            if (board[posi] == piece)
            {
                count++;
                continue;
            }
            else if (board[posi] == EMPTY)
                five += isFive(posi, piece, iDir);
            break;
        }
        for (j = 1; five && j < 6 - i; j++)
        {
            Pos posi = pos + DIRECTION[iDir] * j;
            if (board[posi] == piece)
            {
                count++;
                continue;
            }
            else if (board[posi] == EMPTY)
                five += isFive(posi, piece, iDir);
            break;
        }

        set(pos, EMPTY);
        return five == 2 ? (count == 4 ? OF_TRUE : OF_LONG) : OF_NONE;
    }
    else
        return OF_NONE;
}

bool Board::isOpenThree(Pos pos, Color piece, int iDir)
{
    if (board[pos] != EMPTY)
        return false;
    else if (isFive(pos, piece))
        return false;
    else if (piece == BLACK && isOverline(pos, BLACK))
        return false;
    else if (piece == BLACK || piece == WHITE)
    {
        bool openthree = false;
        set(pos, piece);

        int i, j;
        for (i = 1; i < 5; i++)
        {
            Pos posi = pos - DIRECTION[iDir] * i;
            if (board[posi] == piece)
                continue;
            else if (board[posi] == EMPTY && isOpenFour(posi, piece, iDir) == OF_TRUE && !isDoubleFour(posi, piece) && !isDoubleThree(posi, piece))
                openthree = true;
            break;
        }
        for (j = 1; !openthree && j < 6 - i; j++)
        {
            Pos posi = pos + DIRECTION[iDir] * j;
            if (board[posi] == piece)
                continue;
            else if (board[posi] == EMPTY && isOpenFour(posi, piece, iDir) == OF_TRUE && !isDoubleFour(posi, piece) && !isDoubleThree(posi, piece))
                openthree = true;
            break;
        }

        set(pos, EMPTY);
        return openthree;
    }
    else
        return false;
}

bool Board::isDoubleFour(Pos pos, Color piece)
{
    if (board[pos] != EMPTY)
        return false;
    else if (isFive(pos, piece))
        return false;

    int nFour = 0;
    for (int iDir = 0; iDir < 4; iDir++)
    {
        if (isOpenFour(pos, piece, iDir) == OF_LONG)
            nFour += 2;
        else if (isFour(pos, piece, iDir))
            nFour++;

        if (nFour >= 2)
            return true;
    }

    return false;
}

bool Board::isDoubleThree(Pos pos, Color piece)
{
    if (board[pos] != EMPTY)
        return false;
    else if (isFive(pos, piece))
        return false;

    int nThree = 0;
    for (int iDir = 0; iDir < 4; iDir++)
    {
        if (isOpenThree(pos, piece, iDir))
            nThree++;

        if (nThree >= 2)
            return true;
    }

    return false;
}