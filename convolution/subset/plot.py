import numpy as np
from matplotlib import pyplot as plt


def get_data(file, scale=1.0):
    file = "data/" + file
    raw = open(file, 'r').readlines()
    data = list(map(lambda x: list(map(float, x.split())), raw))
    x, y = zip(*data)
    x = np.array(x)
    y = np.array(y)
    y /= 2**x
    y *= scale
    return x, y * 1e9


def plot(file, scale=1.0):
    x, y = get_data(file, scale)
    plt.plot(x, y, label=file)
    # if file == "data_ntt.txt":
    #     plt.axline((x[9], y[9]), (x[15], y[15]), color="red")
    # plt.axline((x[9+1], y[9+1]), (x[15+1], y[15+1]), color="red")


def make_plot(files, out_file, show=False, large_y_ticks=False):
    my_dpi = 200
    plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    if large_y_ticks:
        plt.yticks(np.arange(0, 301, 30))
    else:
        plt.yticks(np.arange(0, 21, 1))

    plt.xticks(np.arange(0, 31, 1))
    plt.grid(linestyle="--")
    plt.axvline(x=13, linestyle="--")
    plt.axvline(x=18, linestyle="--")
    plt.axvline(x=21, linestyle="--")
    plt.axvline(x=0)
    plt.axhline(y=0)

    for fl in files:
        plot(fl)
    plt.legend()
    if show:
        plt.show()

    plt.ylabel("ns per element")
    plt.xlabel("log_2 n")

    plt.savefig(f"{out_file}.svg")


# if 1:
#     x1, y1 = get_data("data_ntt.txt")
#     x2, y2 = get_data("../A/data_ntt.txt")
#     plt.plot(x1, y2 / y1)
#     plt.show()

# make_plot(["data_r8.txt"], "plot")
# make_plot(["data_r2.txt", "data_r4.txt", "data_r8.txt", "data_r16.txt"], "plot")
make_plot(["data_r2.txt", "data_r4.txt", "data_r8.txt", "data_r16.txt",
           "data_r2_rec.txt", "data_r4_rec.txt", "data_r8_rec.txt", "data_r16_rec.txt"], "plot")
make_plot(["data_sb_conv.txt", "data_sb_conv2.txt"], "plot_cnv", large_y_ticks=True)
