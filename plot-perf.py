import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Plot performance data")
    parser.add_argument(
        "-k",
        type=str,
        default=""
    )
    parser.add_argument(
        "-p",
        type=str,
        default=""
    )
    parser.add_argument(
        "-c",
        type=str,
        default="Torch",
    )
    parser.add_argument(
        "-s",
        type=str,
        default="log",
    )
    return parser.parse_args()

def plot_performance(k, p, c, s):
    df_with = pd.read_csv(f"./{k}-performance.csv")
    df_without = pd.read_csv(f"./{k}-w:o-{p}.csv")

    # x-axis value is from the first column
    x_with = df_with.iloc[:, 0]
    x_without = df_without.iloc[:, 0]
    x_comp = df_with.iloc[:, 0]

    # triton performance is in the last column
    y_with = df_with['Triton']
    y_without = df_without['Triton']
    # comp performance is in the second column frome the back
    y_comp = df_with[c]

    # draw the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_with, y_with, label=f"w-{p}", color="blue")
    plt.plot(x_without, y_without, label=f"wo-{p}", color="red")
    plt.plot(x_comp, y_comp, label=c, color="green")
    plt.xlabel("size")
    plt.ylabel("performance")
    plt.title(f"Performance of {k}-{p}")
    plt.xscale(s)
    # draw 
    plt.legend()
    plt.grid()
    plt.savefig(f"{k}-{p}.png")
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    k = args.k
    p = args.p
    c = args.c
    s = args.s
    plot_performance(k, p, c, s)