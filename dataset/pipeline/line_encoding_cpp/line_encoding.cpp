#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <vector>
#include <array>
#include <memory>
#include <mutex>
#include <string>

namespace py = pybind11;

class LineEncodingEncoder
{
public:
    /// Initialize the compact line encoder.
    LineEncodingEncoder(int length) : length(length)
    {
        assert(length % 2 == 1); // length must be a odd number
        assert(length < Pow3.size());
        initialize_chunk_lut();
    }

    /// Encode a contiguous key with known distances to the nearest walls.
    /// @param key A key is an unsigned integer, where the lower 2*length bits
    /// encodes the state of a line, with each two bits represents one cell:
    ///     1. 0b00: WALL (outside of board)
    ///     2. 0b01: SELF
    ///     3. 0b10: OPPO
    ///     4. 0b11: EMPTY
    uint32_t encode_contiguous(size_t key, int left, int right) const
    {
        key &= (uint64_t(1) << (2 * length)) - 1;
        uint32_t code = boarder_encodings[left][right];
        for (int chunk = 0; chunk < num_chunks; chunk++)
            code += chunk_lut[chunk][(key >> (12 * chunk)) & 0xfff];
        return code;
    }

    /// Mark every encoding reachable by a line whose center is on the board.
    void fill_usage_flags(int8_t *usage_flags) const
    {
        const int half = length / 2;
        for (int left = 0; left <= half; left++)
        {
            for (int right = 0; right <= half; right++)
            {
                fill_usage_flags_recursive(
                    usage_flags,
                    half - right,
                    half + left,
                    boarder_encodings[left][right]);
            }
        }
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
    const int length;
    int num_chunks = 0;
    std::array<std::array<uint32_t, 4096>, 4> chunk_lut{};
    std::array<std::array<uint32_t, 10>, 10> boarder_encodings{};

    static constexpr auto Pow3 = []()
    {
        auto pow3 = std::array<uint32_t, 20>{};
        uint32_t val = 1;
        for (size_t i = 0; i < pow3.size(); i++, val *= 3)
            pow3[i] = val;
        return pow3;
    }();

    void fill_usage_flags_recursive(
        int8_t *usage_flags,
        int line_idx,
        int last_line_idx,
        uint32_t code) const
    {
        if (line_idx > last_line_idx)
        {
            usage_flags[code] = 1;
            return;
        }
        fill_usage_flags_recursive(
            usage_flags, line_idx + 1, last_line_idx, code);
        fill_usage_flags_recursive(
            usage_flags, line_idx + 1, last_line_idx, code + Pow3[line_idx]);
        fill_usage_flags_recursive(
            usage_flags, line_idx + 1, last_line_idx, code + 2 * Pow3[line_idx]);
    }

    void initialize_chunk_lut()
    {
        const int half = length / 2;
        constexpr int CELLS_PER_CHUNK = 6;
        num_chunks = (length + CELLS_PER_CHUNK - 1) / CELLS_PER_CHUNK;
        for (int chunk = 0; chunk < num_chunks; chunk++)
        {
            for (int value = 0; value < 4096; value++)
            {
                uint32_t contribution = 0;
                for (int offset = 0; offset < CELLS_PER_CHUNK; offset++)
                {
                    const int key_idx = chunk * CELLS_PER_CHUNK + offset;
                    if (key_idx >= length)
                        break;
                    const int state = (value >> (2 * offset)) & 0b11;
                    const int digit = state == 0b01 ? 1 : state == 0b10 ? 2
                                                                       : 0;
                    contribution += digit * Pow3[length - 1 - key_idx];
                }
                chunk_lut[chunk][value] = contribution;
            }
        }

        for (int left = 0; left <= half; left++)
        {
            for (int right = 0; right <= half; right++)
            {
                boarder_encodings[left][right] = get_boarder_encoding(left, right);
            }
        }
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

constexpr int MAX_COMPRESSED_LINE_LENGTH = 17;
constexpr int MAX_RAW_LINE_LENGTH = 15;

static void validate_line_length(int line_length, bool raw_code = false)
{
    if (line_length < 1 || line_length % 2 == 0)
        throw std::invalid_argument("the line length must be a positive odd number");
    const int max_length = raw_code ? MAX_RAW_LINE_LENGTH : MAX_COMPRESSED_LINE_LENGTH;
    if (line_length > max_length)
        throw std::invalid_argument(
            std::string("the maximum line length for ") +
            (raw_code ? "raw code is " : "compressed encoding is ") +
            std::to_string(max_length));
}

static std::vector<std::unique_ptr<LineEncodingEncoder>> Encoders;
static std::mutex EncodersMutex;

/// Gets or creates a compact encoder without allocating a 4^L lookup table.
static const LineEncodingEncoder &get_line_encoder(int length)
{
    validate_line_length(length);
    std::lock_guard<std::mutex> lock(EncodersMutex);
    for (const auto &encoder : Encoders)
    {
        if (encoder->line_length() == length)
            return *encoder;
    }
    return *Encoders.emplace_back(std::make_unique<LineEncodingEncoder>(length));
}

/// Rotate right with the given shift amount.
inline uint64_t rotate_right(uint64_t x, int shamt)
{
    // (-shamt & 63) keeps both shift counts in [0, 63]; a plain
    // "x << (64 - shamt)" would shift by 64 (UB) whenever shamt == 0.
    shamt &= 63;
    return (x >> shamt) | (x << (-shamt & 63));
}

/// Validate an output array so results are written in place. With a forcecast
/// (or merely castable/non-contiguous) argument, pybind would write into a
/// temporary copy that is silently discarded, leaving the caller's array
/// untouched; reject such arrays loudly instead.
template <typename T>
static py::array_t<T> check_output_array(const py::array &array, const char *name)
{
    if (!py::isinstance<py::array_t<T>>(array))
        throw std::invalid_argument(std::string(name) + " has incorrect dtype");
    if (!(array.flags() & py::array::c_style))
        throw std::invalid_argument(std::string(name) + " must be C-contiguous");
    if (!array.writeable())
        throw std::invalid_argument(std::string(name) + " must be writeable");
    return py::reinterpret_borrow<py::array_t<T>>(array);
}

/// Get the total number of line encoding of the line length.
int get_total_num_encoding(int line_length)
{
    validate_line_length(line_length);
    return LineEncodingEncoder::max_encoding(line_length) + 1;
}

/// Get the usage flags of each line encoding.
/// @param usage_flags_output usage flags output numpy array, must be initialized to zero.
void get_encoding_usage_flag(
    py::array usage_flags_output,
    int line_length)
{
    auto usage_flags_array = check_output_array<int8_t>(usage_flags_output, "usage_flags_output");
    auto usage_flags = usage_flags_array.mutable_unchecked<1>();
    const auto &encoder = get_line_encoder(line_length);
    if (encoder.total_num_encoding() != (size_t)usage_flags.shape(0))
        throw std::invalid_argument("invalid usage_flags shape");

    encoder.fill_usage_flags(usage_flags_array.mutable_data());
}

template <typename EncodeKey>
static void transform_board_impl(
    const int8_t *board,
    int32_t *line_encoding,
    int H,
    int W,
    int line_length,
    EncodeKey encode_key,
    bool raw_code)
{
    constexpr int MAX_BOARD_SIZE = 32;
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
            set_bit_key(
                x,
                y,
                board[y * W + x],
                board[H * W + y * W + x]);

    // Encode every position and direction.
    const int half = line_length / 2;
    // Same value as total_num_key() - 1, computed without an encoder.
    const uint64_t raw_mask = (uint64_t(1) << (2 * line_length)) - 1;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
        {
            uint64_t key0 = rotate_right(bit_key0[y], 2 * (x - half));
            uint64_t key1 = rotate_right(bit_key1[x], 2 * (y - half));
            uint64_t key2 = rotate_right(bit_key2[x + y], 2 * (x - half));
            uint64_t key3 = rotate_right(bit_key3[MAX_BOARD_SIZE - 1 - x + y], 2 * (x - half));
            const int left0 = std::min(half, x);
            const int right0 = std::min(half, W - 1 - x);
            const int left1 = std::min(half, y);
            const int right1 = std::min(half, H - 1 - y);
            const int left2 = std::min({half, x, H - 1 - y});
            const int right2 = std::min({half, W - 1 - x, y});
            const int left3 = std::min({half, x, y});
            const int right3 = std::min({half, W - 1 - x, H - 1 - y});

            if (raw_code)
            {
                line_encoding[0 * H * W + y * W + x] = uint32_t(key0 & raw_mask);
                line_encoding[1 * H * W + y * W + x] = uint32_t(key1 & raw_mask);
                line_encoding[2 * H * W + y * W + x] = uint32_t(key2 & raw_mask);
                line_encoding[3 * H * W + y * W + x] = uint32_t(key3 & raw_mask);
            }
            else
            {
                line_encoding[0 * H * W + y * W + x] = encode_key(key0, left0, right0);
                line_encoding[1 * H * W + y * W + x] = encode_key(key1, left1, right1);
                line_encoding[2 * H * W + y * W + x] = encode_key(key2, left2, right2);
                line_encoding[3 * H * W + y * W + x] = encode_key(key3, left3, right3);
            }
        }
}

static void transform_board_dispatch(
    const int8_t *board,
    int32_t *line_encoding,
    int H,
    int W,
    int line_length,
    const LineEncodingEncoder *encoder,
    bool raw_code)
{
    if (raw_code)
    {
        transform_board_impl(
            board,
            line_encoding,
            H,
            W,
            line_length,
            [](size_t, int, int)
            { return uint32_t(0); },
            true);
    }
    else
    {
        transform_board_impl(
            board,
            line_encoding,
            H,
            W,
            line_length,
            [encoder](size_t key, int left, int right)
            { return encoder->encode_contiguous(key, left, right); },
            false);
    }
}

/// Transform a board input numpy array to 4 direction line encoding output numpy array.
/// @param board_input Board numpy array of shape [2, H, W]. First/second channel is self/oppo.
/// @param line_encoding_output Line encoding numpy array of shape [4, H, W].
/// @param line_length The length of line to encode.
/// @param raw_code Whether to output raw bit code instead of line encoding.
void transform_board_to_line_encoding(
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> board_input,
    py::array line_encoding_output,
    int line_length,
    bool raw_code = false)
{
    constexpr int MAX_BOARD_SIZE = 32;
    auto board = board_input.unchecked<3>();
    auto line_encoding_array = check_output_array<int32_t>(line_encoding_output, "line_encoding_output");
    auto line_encoding = line_encoding_array.mutable_unchecked<3>();
    int H = (int)board.shape(1), W = (int)board.shape(2);

    if (board.shape(0) != 2)
        throw std::invalid_argument("board shape incorrect, must be [2,H,W]");
    if (line_encoding.shape(0) != 4 || line_encoding.shape(1) != H || line_encoding.shape(2) != W)
        throw std::invalid_argument("line_encoding shape incorrect, must be [4,H,W]");
    validate_line_length(line_length, raw_code);
    // Rotating the 64-bit keys wraps pairs around a 32-cell ring, so segments
    // longer than 32 - line_length/2 would read cells from the far end of the
    // segment instead of WALL. Reject such boards instead of encoding them wrong.
    const int max_board_size = MAX_BOARD_SIZE - line_length / 2;
    if (H > max_board_size || W > max_board_size)
        throw std::invalid_argument("board size must be less or equal to " + std::to_string(max_board_size) +
                                    " for line length " + std::to_string(line_length));
    // The raw code path only masks the bit keys and never reads an encoder.
    const LineEncodingEncoder *encoder =
        raw_code ? nullptr : &get_line_encoder(line_length);
    py::gil_scoped_release release;
    transform_board_dispatch(
        board_input.data(),
        line_encoding_array.mutable_data(),
        H,
        W,
        line_length,
        encoder,
        raw_code);
}

/// Batched variant of transform_board_to_line_encoding.
void transform_boards_to_line_encoding(
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> boards_input,
    py::array line_encodings_output,
    int line_length,
    bool raw_code = false)
{
    constexpr int MAX_BOARD_SIZE = 32;
    auto boards = boards_input.unchecked<4>();
    auto line_encodings_array = check_output_array<int32_t>(line_encodings_output, "line_encodings_output");
    auto line_encodings = line_encodings_array.mutable_unchecked<4>();
    int B = (int)boards.shape(0), H = (int)boards.shape(2), W = (int)boards.shape(3);

    if (boards.shape(1) != 2)
        throw std::invalid_argument("boards shape incorrect, must be [B,2,H,W]");
    if (line_encodings.shape(0) != B || line_encodings.shape(1) != 4 ||
        line_encodings.shape(2) != H || line_encodings.shape(3) != W)
        throw std::invalid_argument("line_encodings shape incorrect, must be [B,4,H,W]");
    validate_line_length(line_length, raw_code);
    // See transform_board_to_line_encoding for why the bound depends on line_length.
    const int max_board_size = MAX_BOARD_SIZE - line_length / 2;
    if (H > max_board_size || W > max_board_size)
        throw std::invalid_argument("board size must be less or equal to " + std::to_string(max_board_size) +
                                    " for line length " + std::to_string(line_length));
    // The raw code path only masks the bit keys and never reads an encoder.
    const LineEncodingEncoder *encoder =
        raw_code ? nullptr : &get_line_encoder(line_length);
    const size_t board_stride = 2 * H * W;
    const size_t encoding_stride = 4 * H * W;
    const int8_t *boards_ptr = boards_input.data();
    int32_t *encodings_ptr = line_encodings_array.mutable_data();
    py::gil_scoped_release release;
    for (int b = 0; b < B; b++)
        transform_board_dispatch(
            boards_ptr + b * board_stride,
            encodings_ptr + b * encoding_stride,
            H,
            W,
            line_length,
            encoder,
            raw_code);
}

template <typename EncodeKey>
static void transform_lines_impl(
    const int8_t *lines,
    int32_t *line_encodings,
    int N,
    int L,
    int line_length,
    EncodeKey encode_key)
{
    const int half = line_length / 2;
    for (int n = 0; n < N; n++)
    {
        uint64_t bit_key = 0; // [RIGHT(MSB) - LEFT(LSB)]
        for (int x = 0; x < L; x++)
        {
            const int8_t value = lines[n * L + x];
            bool is_self = value == 1;
            bool is_oppo = value == 2;
            uint64_t cell_bits = is_self ? 0b01 : is_oppo ? 0b10
                                                              : 0b11;
            bit_key |= cell_bits << (2 * x);
        }

        for (int x = 0; x < L; x++)
        {
            uint64_t key = rotate_right(bit_key, 2 * (x - half));
            const int left = std::min(half, x);
            const int right = std::min(half, L - 1 - x);
            line_encodings[n * L + x] = encode_key(key, left, right);
        }
    }
}

/// Transform batched lines numpy array to line encoding output numpy array.
/// @param lines_input Lines numpy array of shape [N, L]. Elements are in {0,1,2} for empty/self/oppo.
/// @param line_encodings_output Line encoding numpy array of shape [N, L].
/// @param line_length The length of line to encode.
void transform_lines_to_line_encoding(
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> lines_input,
    py::array line_encodings_output,
    int line_length)
{
    constexpr int MAX_BOARD_SIZE = 32;

    auto lines = lines_input.unchecked<2>();
    auto line_encodings_array = check_output_array<int32_t>(line_encodings_output, "line_encodings_output");
    auto line_encodings = line_encodings_array.mutable_unchecked<2>();
    int N = (int)lines.shape(0), L = (int)lines.shape(1);

    if (line_encodings.shape(0) != N || line_encodings.shape(1) != L)
        throw std::invalid_argument("line_encodings shape incorrect, must be [N, L]");
    validate_line_length(line_length);
    const int max_input_length = MAX_BOARD_SIZE - line_length / 2;
    if (L > max_input_length)
        throw std::invalid_argument("input length must be less or equal to " + std::to_string(max_input_length) +
                                    " for line length " + std::to_string(line_length));

    const auto &encoder = get_line_encoder(line_length);
    transform_lines_impl(
        lines_input.data(),
        line_encodings_array.mutable_data(),
        N,
        L,
        line_length,
        [&encoder](size_t key, int left, int right)
        { return encoder.encode_contiguous(key, left, right); });
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
    m.def("transform_boards_to_line_encoding", &transform_boards_to_line_encoding,
          "boards_input"_a,
          "line_encodings_output"_a,
          "line_length"_a,
          "raw_code"_a = false);
    m.def("transform_lines_to_line_encoding", &transform_lines_to_line_encoding,
          "lines_input"_a,
          "line_encodings_output"_a,
          "line_length"_a);
}
