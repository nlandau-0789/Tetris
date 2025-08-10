import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

with open("logs_lr5", "r") as f:
    x = [float(i) for i in f.read().split()]
t = [i for i, j in enumerate(x)]

# Parameters
window_size = 500
step_size = 50  # larger step for quicker animation
bins = 30
frames = (len(x) - window_size) // step_size

# Set up plot
fig, ax = plt.subplots()
hist_container = ax.hist([], bins=bins, range=(np.min(x), np.max(x)))[2]

def update(frame):
    ax.clear()  # clear previous frame

    start = (((frame * step_size) % 4000) + 9000) 
    end = start + window_size
    # if end > len(x):
    #     return  # skip if window goes out of bounds

    window_data = x[start:end]
    # print(window_data)

    # Histogram
    ax.hist(window_data, bins=bins, color='skyblue', edgecolor='black', alpha=1)

    # ax.set_xlim(np.min(x), np.max(x))
    ax.set_title(f"{start}:{end}")
    # ax.set_xlabel("x")
    # ax.set_ylabel("Count")

# Run animation
ani = FuncAnimation(fig, update, frames=frames, interval=200, repeat=False)
plt.tight_layout()
plt.show()