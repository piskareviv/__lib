from matplotlib import pyplot as plt
import numpy as np


def load(file):
    txt = open(file).read()
    data = eval(txt)
    x, dt = list(zip(*data))
    return x, np.array(dt) * 1e6  # convert to ns


plt.yticks(np.arange(0, 51, 1))
plt.xticks(np.arange(5, 31, 1))
plt.grid(linestyle="--")

plt.axvline(x=13, linestyle="--")
plt.axvline(x=18, linestyle="--")
plt.axvline(x=21, linestyle="--")

plt.axvline(x=0)
plt.axhline(y=0)


def fuck(file, scale=1):
    x, data = load(file)
    x = np.exp2(x)
    plt.plot(np.log2(x), data / x * scale, label=file)


fuck("ntt_rec.txt")
fuck("ntt_rec_hrd.txt")
fuck("ntt_rec_hrd_2.txt")
fuck("ntt_rec_hrd_3.txt")
fuck("ntt_rec_x4.txt", 0.25)
fuck("ntt_rec_hrd_x4.txt", 0.25)
fuck("ntt.txt")


plt.legend()
plt.show()
