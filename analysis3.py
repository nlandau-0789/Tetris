import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    """Compute the moving average of a 1D array."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot(*filenames, window_size=250):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for filename in filenames:
        with open(filename, "r") as f:
            v = f.read().split("|")
            v2 = (tuple(float(j) for j in i.split()) for i in v if i.strip())
            y, M, A = zip(*v2)
        x = [(i+1) for i, j in enumerate(y)]
        y_smooth = moving_average(y, window_size)
        M_smooth = moving_average(M, window_size)
        A_smooth = moving_average(A, window_size)
        x_smooth = x[window_size-1:]


        # Plot y_smooth on the left y-axis
        ax1.plot(x_smooth, y_smooth, label=f"y ({filename})")

        # Create a second y-axis for M_smooth and A_smooth
        # ax2.plot(x_smooth, M_smooth, label=f"max_rew ({filename})")
        # ax2.plot(x_smooth, A_smooth, label=f"avg_rew ({filename})")

    ax1.tick_params(axis='y')
    ax2.tick_params(axis='y')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.show()

plot("logs_scores_eval1", "logs_scores_eval2", "logs_scores_eval3", window_size=20)
