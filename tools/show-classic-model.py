import argparse
import lz4.frame
import struct

N = 14


def combine_number(n, m):
    r = 1
    for i in range(n, m + n):
        r *= i
    for i in range(2, m + 1):
        r //= i
    return r


def show_values(f, name, num_values, num_per_line):
    print(name.center(80, "-"))
    for i in range(num_values):
        (v,) = struct.unpack("<h", f.read(2))
        print(f"{v:5} ", end="\n" if (i + 1) % num_per_line == 0 else "")
    print()


def show_policy(f, name, num_values, num_per_line):
    print(name.center(80, "-"))
    for i in range(num_values):
        (pb,) = struct.unpack("<h", f.read(2))
        (pw,) = struct.unpack("<h", f.read(2))
        print(f"[{pb:4},{pw:4}]   ", end="\n" if (i + 1) % num_per_line == 0 else "")
    print()


def show_model():
    parser = argparse.ArgumentParser(description="Show model values")
    parser.add_argument("model", type=str, help="Model path")
    args = parser.parse_args()

    PCODE_NB = combine_number(N, 4)
    THREAT_NB = 2**11

    with lz4.frame.open(args.model, "rb") as f:
        f.read(8)  # Discard scaling factor

        show_values(f, "eval freestyle", PCODE_NB, N)
        show_values(f, "eval standard", PCODE_NB, N)
        show_values(f, "eval renju black", PCODE_NB, N)
        show_values(f, "eval renju white", PCODE_NB, N)

        show_values(f, "threat freestyle", THREAT_NB, 16)
        show_values(f, "threat standard", THREAT_NB, 16)
        show_values(f, "threat renju black", THREAT_NB, 16)
        show_values(f, "threat renju white", THREAT_NB, 16)

        show_policy(f, "policy score freestyle", PCODE_NB, N // 2)
        show_policy(f, "policy score standard", PCODE_NB, N // 2)
        show_policy(f, "policy score renju black", PCODE_NB, N // 2)
        show_policy(f, "policy score renju white", PCODE_NB, N // 2)


if __name__ == "__main__":
    show_model()
