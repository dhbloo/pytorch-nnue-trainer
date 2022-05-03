#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <cassert>
#include <vector>
#include <array>

namespace py = pybind11;

class LineEncodingTable
{
public:
    /// Initialize the line encoding table.
    LineEncodingTable(int length) : length(length)
    {
        assert(length % 2 == 1); // length must be a odd number
        assert(length < Pow3.size());

        size_t pow4_len = 1;
        for (int l = 0; l < length; l++)
            pow4_len *= 4;

        encodings.resize(pow4_len);
        for (size_t key = 0; key < pow4_len; key++)
            encodings[key] = get_key_encoding(key);
    }

    /// Query the encoding of a line key.
    /// @param key A key is an unsigned integer, where the lower 2*length bits
    /// encodes the state of a line, with each two bits represents one cell:
    ///     1. 0b00: WALL (outside of board)
    ///     2. 0b01: SELF
    ///     3. 0b10: OPPO
    ///     4. 0b11: EMPTY
    uint32_t operator[](size_t key) const
    {
        return encodings[key & (encodings.size() - 1)];
    }

    /// Get the length of a line.
    int line_length() const
    {
        return length;
    }

    /// The maximum possible encoding that might occur.
    uint32_t max_encoding() const
    {
        return max_encoding(length);
    }

    /// The total number of encodings.
    size_t total_num_encoding() const
    {
        return max_encoding() + 1;
    }

    /// Get the maximum possible encoding from line of length that might occur.
    static uint32_t max_encoding(int len)
    {
        const int half = len / 2;
        uint32_t code = 2 * Pow3[len];
        for (int i = 0; i <= half; i++)
            code += 2 * Pow3[i];
        for (int i = half + 2; i < len; i++)
            code += 1 * Pow3[i];
        return code;
    }

private:
    enum ColorFlag
    {
        SELF,
        OPPO,
        EMPTY,
        WALL
    };

    const int length;
    std::vector<uint32_t> encodings;

    static constexpr auto Pow3 = []()
    {
        auto pow3 = std::array<uint32_t, 20>{};
        uint32_t val = 1;
        for (size_t i = 0; i < pow3.size(); i++, val *= 3)
            pow3[i] = val;
        return pow3;
    }();

    /// Convert a key to its corrsponding line.
    void key_to_line(size_t key, ColorFlag *line) const
    {
        for (int i = 0; i < length; i++)
        {
            switch ((key >> (2 * i)) & 0b11)
            {
            case 0b00:
                line[i] = WALL;
                break;
            case 0b01:
                line[i] = SELF;
                break;
            case 0b10:
                line[i] = OPPO;
                break;
            case 0b11:
                line[i] = EMPTY;
                break;
            }
        }
    }

    /// Get an line encoding with the given key.
    /// @param key The corrsponding key of a line, in range [0, 4^length).
    uint32_t get_key_encoding(size_t key) const
    {
        const int half = length / 2;
        std::vector<ColorFlag> line;
        line.resize(length);
        key_to_line(key, line.data());

        int left = 0, right = 0;
        for (int i = half + 1; i < length; i++)
        {
            if (line[i] == WALL)
                break;
            left++;
        }
        for (int i = half - 1; i >= 0; i--)
        {
            if (line[i] == WALL)
                break;
            right++;
        }

        uint32_t code = get_boarder_encoding(left, right);
        for (int i = half - right; i <= half + left; i++)
        {
            switch (line[i])
            {
            case SELF:
                code += 1 * Pow3[i];
                break;
            case OPPO:
                code += 2 * Pow3[i];
                break;
            default:
                break; // Empty: code += 0 * Pow3[i];
            }
        }

        return code;
    }

    /// Get an empty line encoding with the given boarder distance.
    /// @param left The distance to the left boarder, in range [0, length/2].
    /// @param right The distance to the right boarder, in range [0, length/2].
    uint32_t get_boarder_encoding(int left, int right) const
    {
        const int half = length / 2;
        assert(0 <= left && left <= half);
        assert(0 <= right && right <= half);

        if (left == half && right == left)
            return 0;
        else if (right == half) // (left < half)
        {
            uint32_t code = 2 * Pow3[length];
            int left_dist = half - left;
            for (int i = 1; i < left_dist; i++)
                code += 1 * Pow3[length - i];
            return code;
        }
        else // (right < half && left <= half)
        {
            uint32_t code = 1 * Pow3[length];
            int left_dist = half - left;
            int right_dist = half - right;
            int right_twos = std::min(left_dist, right_dist);
            int left_twos = std::min(left_dist, right_dist - 1);

            for (int i = 0; i < right_twos; i++)
                code += 2 * Pow3[i];
            for (int i = 0; i < left_twos; i++)
                code += 2 * Pow3[length - 1 - i];

            for (int i = right_twos; i < right_dist - 1; i++)
                code += 1 * Pow3[i];
            for (int i = left_twos; i < left_dist - 1; i++)
                code += 1 * Pow3[length - 1 - i];

            return code;
        }
    }
};

std::vector<LineEncodingTable> EncodingTables;

/// Gets existing encoding table or create a new one.
const LineEncodingTable &get_line_encoding_table(int length)
{
    for (const auto &table : EncodingTables)
    {
        if (table.line_length() == length)
            return table;
    }

    if (length % 2 != 1)
        throw std::invalid_argument("line length must be a odd number");
    if (length >= 20)
        throw std::invalid_argument("max supported line length is 19");

    return EncodingTables.emplace_back(length);
}

/// Right logical shift function that supports negetive shamt.
inline uint64_t right_shift(uint64_t x, int shamt)
{
    return shamt < 0 ? x << (-shamt) : x >> shamt;
}

/// Get the total number of line encoding of the line length.
int get_total_num_encoding(int line_length)
{
    return LineEncodingTable::max_encoding(line_length) + 1;
}

/// Transform a board input numpy array to 4 direction line encoding output numpy array
/// @param board_input Board numpy array of shape [2, H, W].
/// @param line_encoding_output Line encoding numpy array of shape [4, H, W].
void transform_board_to_line_encoding(
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> board_input,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> line_encoding_output,
    int line_length)
{
    constexpr int MAX_BOARD_SIZE = 32;

    auto board = board_input.unchecked<3>();
    auto line_encoding = line_encoding_output.mutable_unchecked<3>();
    int H = (int)board.shape(1), W = (int)board.shape(2);

    // Check input legality
    if (board.shape(0) != 2)
        throw std::invalid_argument("board shape incorrect, must be [2,H,W]");
    if (line_encoding.shape(0) != 4 || line_encoding.shape(1) != H || line_encoding.shape(2) != W)
        throw std::invalid_argument("line_encoding shape incorrect, must be [4,H,W]");
    if (H > MAX_BOARD_SIZE || W > MAX_BOARD_SIZE)
        throw std::invalid_argument("board size must be less or equal to " + std::to_string(MAX_BOARD_SIZE));

    const auto &encoding_table = get_line_encoding_table(line_length);

    // Initialize all bit key to 0b00 (WALL).
    uint64_t bit_key0[MAX_BOARD_SIZE] = {0};         // [RIGHT(MSB) - LEFT(LSB)]
    uint64_t bit_key1[MAX_BOARD_SIZE] = {0};         // [DOWN(MSB) - UP(LSB)]
    uint64_t bit_key2[MAX_BOARD_SIZE * 2 - 1] = {0}; // [UP_RIGHT(MSB) - DOWN_LEFT(LSB)]
    uint64_t bit_key3[MAX_BOARD_SIZE * 2 - 1] = {0}; // [DOWN_RIGHT(MSB) - UP_LEFT(LSB)]

    auto set_bit_key = [&](int x, int y, bool is_self, bool is_oppo)
    {
        uint64_t cell_bits = is_self ? 0b01 : is_oppo ? 0b10
                                                      : 0b11;

        bit_key0[y] |= cell_bits << (2 * x);
        bit_key1[x] |= cell_bits << (2 * y);
        bit_key2[x + y] |= cell_bits << (2 * x);
        bit_key3[MAX_BOARD_SIZE - 1 - x + y] |= cell_bits << (2 * x);
    };

    // Set bit keys
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            set_bit_key(x, y, board(0, y, x), board(1, y, x));

    // Query line encoding table for each pos and direction
    const int half = line_length / 2;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
        {
            uint64_t key0 = right_shift(bit_key0[y], 2 * (x - half));
            uint64_t key1 = right_shift(bit_key1[x], 2 * (y - half));
            uint64_t key2 = right_shift(bit_key2[x + y], 2 * (x - half));
            uint64_t key3 = right_shift(bit_key3[MAX_BOARD_SIZE - 1 - x + y], 2 * (x - half));

            line_encoding(0, y, x) = encoding_table[key0];
            line_encoding(1, y, x) = encoding_table[key1];
            line_encoding(2, y, x) = encoding_table[key2];
            line_encoding(3, y, x) = encoding_table[key3];
        }
}

PYBIND11_MODULE(line_encoding_cpp, m)
{
    m.doc() = "Transform board input to line encoding";
    m.def("get_total_num_encoding", &get_total_num_encoding);
    m.def("transform_board_to_line_encoding", &transform_board_to_line_encoding);
}