from os import system
import re
import math
import sys

data = []

system("g++ bench_run.cpp -O2 -I.. -o bench_run -DLOCAL -std=c++20")

print("compiled", file=sys.stderr, flush=True)
for i in range(5, 30 - 1):
    cnt = int(math.ceil(1e9 / 2**i / i))
    cnt = max(cnt, 10)

    system(f"taskset -c 3 perf stat -d ./bench_run {i} {cnt} 2> tmp.txt")
    # system(f"taskset -c 3 perf stat -d bash cum.sh _ {i} {cnt} 2> tmp.txt")

    s = open("tmp.txt").read()
    mt = re.search("([\d\.]+)\smsec\stask-clock", s.replace(",", ""))
    tm = float(mt.groups()[0])
    data.append([i, tm / cnt])

    print(i, tm, cnt, round(tm/cnt, 5), sep="\t", file=sys.stderr, flush=True)

print(data)
