from sympy import factorint

def fft_input_permutation(n: int):
    factors = factorint(n)
    if not set(factors).issubset({2, 3}):
        raise ValueError("n must be of the form 2*3^a*2^b")
    if factors.get(2, 0) < 1:
        raise ValueError("n must be even")

    a = factors.get(3, 0)
    b = factors.get(2, 0) - 1

    index = [0] * n
    length = 1
    index[0] = 0

    shift = n >> 1
    for i in range(length):
        index[length + i] = index[i] + shift
    length *= 2

    for _ in range(a):
        shift //= 3
        base_len = length
        for i in range(base_len):
            index[length + i] = index[i] + shift
        length += base_len
        for i in range(base_len):
            index[length + i] = index[i] + 2*shift
        length += base_len

    for _ in range(b):
        shift //= 2
        base_len = length
        for i in range(base_len):
            index[length + i] = index[i] + shift
        length += base_len

    return index


def test_fft_input_permutation():
    for n in [12, 16, 24, 36]:
        print(f"n={n}: {fft_input_permutation(n)}")


def main():
    test_fft_input_permutation()


if __name__ == "__main__":
    main()
