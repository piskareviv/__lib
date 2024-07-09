from os import system
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("l", nargs="?", type=int, default=3)
parser.add_argument("r", nargs="?", type=int, default=26)
args = parser.parse_args()
L, R = args.l, args.r


def bench(program, filename, rng):
    filename = "data/" + filename
    print(program, filename, rng, flush=True)
    with open(filename, 'w') as f:
        for i in rng:
            cnt = max(5, int(1e7 / 2**i))
            system(f"taskset -c 3 ./run {i} {cnt} > tmp.txt")
            tm = float(open("tmp.txt", 'r').read())
            print(i, f"{tm / cnt:.12f}", file=f, flush=True)
            print(f"{tm / cnt:.12f}", "  ", i, cnt, tm, flush=True)
    print()


system(f"g++ run.cpp -O2 -std=c++20 -o run")
bench("run", "data_sb_conv.txt", range(L, R))


# system(f"g++ run.cpp -O2 -std=c++20 -o run")
# bench("run", "data_r8.txt", range(L, R))

# system(f"g++ run.cpp -O2 -DRADIX_16 -std=c++20 -o run")
# bench("run", "data_r16.txt", range(L, R))

# system(f"g++ run.cpp -O2 -DRADIX_4 -std=c++20 -o run")
# bench("run", "data_r4.txt", range(L, R))

# system(f"g++ run.cpp -O2 -DRADIX_2 -std=c++20 -o run")
# bench("run", "data_r2.txt", range(L, R))


# system(f"g++ run.cpp -O2 -DREC -std=c++20 -o run")
# bench("run", "data_r8_rec.txt", range(L, R))

# system(f"g++ run.cpp -O2 -DREC -DRADIX_16 -std=c++20 -o run")
# bench("run", "data_r16_rec.txt", range(L, R))

# system(f"g++ run.cpp -O2 -DREC -DRADIX_4 -std=c++20 -o run")
# bench("run", "data_r4_rec.txt", range(L, R))

# system(f"g++ run.cpp -O2 -DREC -DRADIX_2 -std=c++20 -o run")
# bench("run", "data_r2_rec.txt", range(L, R))


system(f"rm tmp.txt")
