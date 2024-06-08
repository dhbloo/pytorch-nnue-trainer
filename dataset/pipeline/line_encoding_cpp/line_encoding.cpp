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

    /// Get the total number of keys.
    size_t total_num_key() const
    {
        return encodings.size();
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
            int line_idx = length - 1 - i;
            switch ((key >> (2 * i)) & 0b11)
            {
            case 0b00:
                line[line_idx] = WALL;
                break;
            case 0b01:
                line[line_idx] = SELF;
                break;
            case 0b10:
                line[line_idx] = OPPO;
                break;
            case 0b11:
                line[line_idx] = EMPTY;
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

        if (left == half && right == half)
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

static std::vector<LineEncodingTable> EncodingTables;

/// Gets existing encoding table or create a new one.
static const LineEncodingTable &get_line_encoding_table(int length)
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

/// Rotate right with the given shift amount.
inline uint64_t rotate_right(uint64_t x, int shamt)
{
    shamt &= 63;
    return (x << (64 - shamt)) | (x >> shamt);
}

/// Get the total number of line encoding of the line length.
int get_total_num_encoding(int line_length)
{
    return LineEncodingTable::max_encoding(line_length) + 1;
}

/// Get the usage flags of each line encoding.
/// @param usage_flags_output usage flags output numpy array, must be initialized to zero.
void get_encoding_usage_flag(
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> usage_flags_output,
    int line_length)
{
    auto usage_flags = usage_flags_output.mutable_unchecked<1>();
    const auto &encoding_table = get_line_encoding_table(line_length);
    if (encoding_table.total_num_encoding() != (size_t)usage_flags.shape(0))
        throw std::invalid_argument("invalid usage_flags shape");

    const int half = line_length / 2;
    for (size_t key = 0; key < encoding_table.total_num_key(); key++)
    {
        uint64_t center_cell_bits = (key >> (2 * half)) & 0b11;
        if (center_cell_bits != 0b00)
            usage_flags[encoding_table[key]] = 1;
    }
}

/// Transform a board input numpy array to 4 direction line encoding output numpy array.
/// @param board_input Board numpy array of shape [2, H, W]. First/second channel is self/oppo.
/// @param line_encoding_output Line encoding numpy array of shape [4, H, W].
/// @param line_length The length of line to encode.
/// @param raw_code Whether to output raw bit code instead of line encoding.
void transform_board_to_line_encoding(
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> board_input,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> line_encoding_output,
    int line_length,
    bool raw_code = false)
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
    if (line_length % 2 == 0)
        throw std::invalid_argument("the line length must be an odd number");
    if (raw_code && line_length > 15)
        throw std::invalid_argument("the maximum line length for raw code is 15");

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
            uint64_t key0 = rotate_right(bit_key0[y], 2 * (x - half));
            uint64_t key1 = rotate_right(bit_key1[x], 2 * (y - half));
            uint64_t key2 = rotate_right(bit_key2[x + y], 2 * (x - half));
            uint64_t key3 = rotate_right(bit_key3[MAX_BOARD_SIZE - 1 - x + y], 2 * (x - half));

            if (raw_code)
            {
                uint64_t mask = encoding_table.total_num_key() - 1;
                line_encoding(0, y, x) = uint32_t(key0 & mask);
                line_encoding(1, y, x) = uint32_t(key1 & mask);
                line_encoding(2, y, x) = uint32_t(key2 & mask);
                line_encoding(3, y, x) = uint32_t(key3 & mask);
            }
            else
            {
                line_encoding(0, y, x) = encoding_table[key0];
                line_encoding(1, y, x) = encoding_table[key1];
                line_encoding(2, y, x) = encoding_table[key2];
                line_encoding(3, y, x) = encoding_table[key3];
            }
        }
}

/// Transform batched lines numpy array to line encoding output numpy array.
/// @param lines_input Lines numpy array of shape [N, L]. Elements are in {0,1,2} for empty/self/oppo.
/// @param line_encodings_output Line encoding numpy array of shape [N, L].
/// @param line_length The length of line to encode.
void transform_lines_to_line_encoding(
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> lines_input,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> line_encodings_output,
    int line_length)
{
    constexpr int MAX_BOARD_SIZE = 32;

    auto lines = lines_input.unchecked<2>();
    auto line_encodings = line_encodings_output.mutable_unchecked<2>();
    int N = (int)lines.shape(0), L = (int)lines.shape(1);

    // Check input legality
    if (L > MAX_BOARD_SIZE)
        throw std::invalid_argument("input length must be less or equal to " + std::to_string(MAX_BOARD_SIZE));
    if (line_encodings.shape(0) != N || line_encodings.shape(1) != L)
        throw std::invalid_argument("line_encodings shape incorrect, must be [N, L]");
    if (line_length % 2 == 0)
        throw std::invalid_argument("the line length must be an odd number");

    const auto &encoding_table = get_line_encoding_table(line_length);
    const int half = line_length / 2;

    // Iterate over all lines
    for (int n = 0; n < N; n++)
    {
        // Initialize bit key to 0b00 (WALL).
        uint64_t bit_key = 0; // [RIGHT(MSB) - LEFT(LSB)]

        // Set bits in bit key
        for (int x = 0; x < L; x++)
        {
            bool is_self = lines(n, x) == 1;
            bool is_oppo = lines(n, x) == 2;
            uint64_t cell_bits = is_self ? 0b01 : is_oppo ? 0b10 : 0b11;
            bit_key |= cell_bits << (2 * x);
        }

        // Query line encoding table for each pos
        for (int x = 0; x < L; x++)
        {
            uint64_t key = rotate_right(bit_key, 2 * (x - half));
            line_encodings(n, x) = encoding_table[key];
        }
    }
}


using namespace py::literals;

PYBIND11_MODULE(line_encoding_cpp, m)
{
    m.doc() = "Transform board input to line encoding";
    m.def("get_total_num_encoding", &get_total_num_encoding, "line_length"_a);
    m.def("get_encoding_usage_flag", &get_encoding_usage_flag,
          "usage_flags_output"_a,
          "line_length"_a);
    m.def("transform_board_to_line_encoding", &transform_board_to_line_encoding,
          "board_input"_a,
          "line_encoding_output"_a,
          "line_length"_a,
          "raw_code"_a = false);
    m.def("transform_lines_to_line_encoding", &transform_lines_to_line_encoding,
          "lines_input"_a,
          "line_encodings_output"_a,
          "line_length"_a);
}