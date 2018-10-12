#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import time

def demo(a):
    plt.cla()
    y = [xt*a/10.0+1 for xt in x]
    ax.set_ylim([0,15])
    ax.plot(x,y)

if __name__ == '__main__':
    plt.ion()
    fig, ax = plt.subplots()
    x = range(5)
    for a in range(1,400):
        demo(a)
        plt.pause(0.01)
        plt.draw()