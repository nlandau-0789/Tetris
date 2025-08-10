import numpy as np
import matplotlib.pyplot as plt

with open("logs_lr", "r") as f:
    y1 = [float(i) for i in f.read().split()]
with open("logs_lr2", "r") as f:
    y2 = [float(i) for i in f.read().split()]
with open("logs_lr3", "r") as f:
    y3 = [float(i) for i in f.read().split()]
with open("logs_lr4", "r") as f:
    y4 = [float(i) for i in f.read().split()]
with open("logs_lr5", "r") as f:
    y5 = [float(i) for i in f.read().split()]
x1 = [500*(i+1) for i, j in enumerate(y1)]
x2 = [500*(i+1) for i, j in enumerate(y2)]
x3 = [500*(i+1) for i, j in enumerate(y3)]
x4 = [100*(i+1) for i, j in enumerate(y4)]
x5 = [i+1 for i, j in enumerate(y5)]




y6 = []
x6 = []
for i in range(0, len(y5), 500):
    y6.append(np.mean(y5[i:i+500]))
    x6.append(np.max(x5[i:i+500]))
    


plt.figure()
# plt.scatter(x1, y1)
# plt.scatter(x2, y2)
plt.scatter(x3, y3)
# plt.scatter(x4, y4)
plt.scatter(x5, y5)
plt.scatter(x6, y6)
plt.show()