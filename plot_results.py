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
            props[key] = value
        else:
            metric_values.append(float(line))
    return props, metric_values


def plot_metric_values(acc, prec, rec):
    fig, ax = plt.subplots()
    ax.plot(acc, label="Accuracy")
    ax.plot(prec, label="Precision")
    ax.plot(rec, label="Recall")
    # plt.title("Perceptron - CountMin | log_buckets = 17, hashes = 3, ngram = 3, learning-rate = 0.8, windows = 100", y=0.8)
    plt.legend(loc="lower right")
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.set_xlabel("windows")
    ax.set_ylabel("metric value")
    plt.show()


if __name__ == "__main__":
    outfile1 = sys.argv[1]
    outfile2 = sys.argv[2]
    outfile3 = sys.argv[3]

    print(f"reading outfile `{outfile1}`")
    with open(outfile1, "r") as f1:
        props1, accuracy = read_outfile(f1)

    print("props", props1)

    print(f"reading outfile `{outfile2}`")
    with open(outfile2, "r") as f2:
        props2, precision = read_outfile(f2)

    print("props", props2)

    print(f"reading outfile `{outfile3}`")
    with open(outfile3, "r") as f3:
        props3, recall = read_outfile(f3)

    print("props", props3)

    plot_metric_values(accuracy, precision, recall)
