import matplotlib.pyplot as plt
import numpy as np

def draw_line_chart(x, ys, legends, path = 'dd.png', colors = None, xlabel = None, ylabel = None):
    plt.clf()
    for i, (y, l) in enumerate(zip(ys, legends)):
        if colors is not None:
            plt.plot(x[:len(y)], y, colors[i], label = l)
        else:
            plt.plot(x[:len(y)], y, label = l)
    plt.legend()
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.savefig(path)
