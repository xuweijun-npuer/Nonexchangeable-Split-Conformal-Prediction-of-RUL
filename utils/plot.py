import matplotlib.pyplot as plt
import numpy as np

def plot_single_intervals(intervals, single_point_predictions, ground_truth_rul, color, label):
    sorted_ground_truth_rul_index = ground_truth_rul.argsort(axis=0, kind='quicksort')[::-1].reshape(-1)
    #sorted_ground_truth_rul_index = ground_truth_rul.argsort(axis=0).reshape(-1)
    sorted_ground_truth_rul = ground_truth_rul[sorted_ground_truth_rul_index]
    lower = intervals[0].reshape(-1)
    upper = intervals[1].reshape(-1)
    sorted_lower = lower[sorted_ground_truth_rul_index]
    sorted_upper = upper[sorted_ground_truth_rul_index]

    plt.figure(figsize=(12, 10))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 25
    plt.fill_between(range(len(sorted_ground_truth_rul)), sorted_lower, sorted_upper, color=color, alpha=1, label=label)
    plt.plot(range(len(sorted_ground_truth_rul)), sorted_ground_truth_rul, 'ok', label="Groundtruth RULs", alpha=0.6)
    plt.plot(range(len(sorted_ground_truth_rul)), single_point_predictions[sorted_ground_truth_rul_index], '--k',
             label="Single-point RUL predictions", alpha=1)
    plt.ylim([0, 250])
    plt.xlabel('Test unit with increasing RUL')
    plt.ylabel('Remaining useful life')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(fname = "../figures/" + label + ".eps", dpi = 300, format="eps")
    plt.savefig(fname = "../figures/" + label + ".svg", dpi = 300, format="svg")
    plt.show()

def plot_two_intervals(intervals1, intervals2, single_point_predictions, ground_truth_rul, color, label):
    sorted_ground_truth_rul_index = ground_truth_rul.argsort(axis=0, kind='quicksort')[::-1].reshape(-1)
    #sorted_ground_truth_rul_index = ground_truth_rul.argsort(axis=0).reshape(-1)
    sorted_ground_truth_rul = ground_truth_rul[sorted_ground_truth_rul_index]
    lower1 = intervals1[0].reshape(-1)
    upper1 = intervals1[1].reshape(-1)
    sorted_lower1 = lower1[sorted_ground_truth_rul_index]
    sorted_upper1 = upper1[sorted_ground_truth_rul_index]
    lower2 = intervals2[0].reshape(-1)
    upper2 = intervals2[1].reshape(-1)
    sorted_lower2 = lower2[sorted_ground_truth_rul_index]
    sorted_upper2 = upper2[sorted_ground_truth_rul_index]

    plt.figure(figsize=(12, 10))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 25
    plt.fill_between(range(len(sorted_ground_truth_rul)), sorted_lower1, sorted_upper1, color=color[0], alpha=1,
                     label=label[0])
    plt.fill_between(range(len(sorted_ground_truth_rul)), sorted_lower2, sorted_upper2, color=color[1], alpha=1,
                     label=label[1])
    plt.fill_between(range(len(sorted_ground_truth_rul)), np.maximum(sorted_lower1, sorted_lower2),
                     np.minimum(sorted_upper1, sorted_upper2), alpha=1, label=label[2])
    plt.plot(range(len(sorted_ground_truth_rul)), sorted_ground_truth_rul, 'ok', label="Groundtruth RULs",
                     alpha=0.6)
    plt.plot(range(len(sorted_ground_truth_rul)), single_point_predictions[sorted_ground_truth_rul_index], '--k',
             label="Single-point RUL predictions", alpha=1)
    plt.ylim([0, 250])
    plt.xlabel('Test unit with increasing RUL')
    plt.ylabel('Remaining useful life')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(fname = "../figures/" + label[0] + "-" + label[1] + ".eps", dpi = 300, format="eps")
    plt.savefig(fname = "../figures/" + label[0] + "-" + label[1] + ".svg", dpi = 300, format="svg")
    plt.show()
