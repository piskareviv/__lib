from matplotlib import pyplot as plt
import numpy as np


def load(file):
    txt = open(file).read()
    data = eval(txt)
    x, dt = list(zip(*data))
    return x, np.array(dt) * 1e6  # convert to ns


x, wht_radix4 = load("bench_data_fwht_radix4.txt")
x, wht_radix8 = load("bench_data_fwht_radix8.txt")
x, wht_wtf = load("bench_data_fwht_wtf.txt")
x, wht_wtf2 = load("bench_data_fwht_wtf2.txt")
x, wht_rec8_radix8 = load("bench_data_fwht_rec8_radix8.txt")
x, wht_rec4_radix8 = load("bench_data_fwht_rec4_radix8.txt")

x0, ssa_radix2_ntt = load("bench_data_ssa_radix2_ntt.txt")
x, ntt_radix4 = load("bench_data_ntt_radix4.txt")

x0 = np.exp2(x0)
x = np.exp2(x)

plt.yticks(np.arange(0, 51, 1))
plt.xticks(np.arange(5, 31, 1))
plt.grid(linestyle="--")

plt.axvline(x=13, linestyle="--")
plt.axvline(x=18, linestyle="--")
plt.axvline(x=21, linestyle="--")

# plt.axvline(x=0)
plt.axhline(y=0)

plt.plot(np.log2(x), wht_radix4 / x, label="wht_radix4 per element")
plt.plot(np.log2(x), wht_radix8 / x, label="wht_radix8 per element")
plt.plot(np.log2(x), wht_wtf / x, label="wht_wtf per element")
plt.plot(np.log2(x), wht_wtf2 / x, label="wht_wtf2 per element")
plt.plot(np.log2(x), wht_rec8_radix8 / x, label="wht_rec8_radix8 per element")
plt.plot(np.log2(x), wht_rec4_radix8 / x, label="wht_rec4_radix8 per element")
plt.plot(np.log2(x), ntt_radix4 / x, label="ntt_radix4 per element")
plt.plot(np.log2(x0), ssa_radix2_ntt / x0, label="ssa_radix2_ntt per element")

# plt.plot([-14, 7, 28, 49], [-1, 2, 5, 8])

#

plt.legend()
plt.show()
