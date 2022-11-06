# Copyright 2022 BDAP team.
#
# Author: Laurens Devos
# Version: 0.1

import sys
import matplotlib.pyplot as plt

def read_outfile(outfile):
    props = {}
    metric_values = []
    for line in outfile:
        if "=" in line:
            key, value = line.split("=")
            props[key] = int(value)
        else:
            metric_values.append(float(line))
    return props, metric_values

def plot_metric_values(metric_values):
    fig, ax = plt.subplots()
    ax.plot(metric_values)
    ax.set_xlabel("time")
    ax.set_ylabel("metric value")
    plt.show()

if __name__ == "__main__":
    outfile = sys.argv[1]

    print(f"reading outfile `{outfile}`")
    with open(outfile, "r") as f:
        props, metric_values = read_outfile(f)

    print("props", props)
    plot_metric_values(metric_values)

