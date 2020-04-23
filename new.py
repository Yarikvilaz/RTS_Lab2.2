import numpy as np
from random import uniform
from datetime import datetime
from math import sin, cos, pi
import matplotlib.pyplot as plt


def generate(n_harmonic=12, lim_freq=1800, n_ticks=64):
    x_array = [0] * n_ticks
    for t in range(n_ticks):
        for i in range(n_harmonic):
            x_array[t] += uniform(0, 1) * sin(lim_freq * (i / n_harmonic) * t + uniform(0, 1))
    return x_array


def count_fx(x):
    n_ticks = 64
    half_of_ticks = n_ticks // 2

    w = np.zeros((half_of_ticks, half_of_ticks))
    w_pn = np.zeros(n_ticks)
    f_i = np.zeros(half_of_ticks)
    f_ii = np.zeros(half_of_ticks)
    f = np.zeros(n_ticks)

    w_fill_expression = lambda a, b: cos(4 * pi / n_ticks * a * b) + sin(4 * pi / n_ticks * a * b)
    w_pn_fill_expression = lambda a: cos(2 * pi / n_ticks * a) + sin(2 * pi / n_ticks * a)

    for i in range(half_of_ticks):
        for j in range(half_of_ticks):
            w[i][j] = w_fill_expression(i, j)

    for i in range(n_ticks):
        w_pn[i] = w_pn_fill_expression(i)

    for i in range(half_of_ticks):
        for j in range(half_of_ticks):
            f_ii[i] += x[2 * j] * w[i][j]
            f_i[i] += x[2 * j + 1] * w[i][j]

    for i in range(n_ticks):
        if i < half_of_ticks:
            f[i] += f_ii[i] + w_pn[i] * f_i[i]
        else:
            f[i] += f_ii[i - half_of_ticks] - w_pn[i] * f_i[i - half_of_ticks]
    return f


def count_fx_prev(x_t):
    fx = []
    for i in range(len(x_t)):
        s = 0
        for j in range(len(x_t)):
            s += x_t[j] * complex(cos(-2 * pi * j * i / len(x_t)), sin(-2 * pi * j * i / len(x_t)))
        fx.append(s)
    return fx


def show(fx):
    plt.plot(fx, label='Fx')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # show(count_fx(generate()))
    x_t = generate()
    t1 = datetime.now()
    count_fx_prev(x_t)
    t2 = datetime.now()
    count_fx(x_t)
    t3 = datetime.now()
    print(f"ДФТ : {t2-t1}")
    print(f"ФФТ : {t3-t2}")