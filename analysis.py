import numpy as np
import matplotlib.pyplot as plt

plt.figure()

def moving_average(data, window_size):
    """Compute the moving average of a 1D array."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot(filename, window_size=250):
    with open(filename, "r") as f:
        y = [float(i) for i in f.read().split()]
    x = [(i+1) for i, j in enumerate(y)]
    y_smooth = moving_average(y, window_size)
    x_smooth = x[window_size-1:]
    # plt.scatter(x, y, alpha=.01)
    plt.plot(x_smooth, y_smooth, label=filename)


# plot("logs_depth5")
# plot("logs_lr6")
# plot("logs_lr8")
plot("logs_lr9")
plot("logs_lr13")
plot("logs_lr12")
plt.show()