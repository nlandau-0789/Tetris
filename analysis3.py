with open("logs", "r") as file:
    lines = file.readlines()
    v = []
    for line in lines:
        if (line.strip()):
            v.append(float(line.strip().split(" = ")[-1]))

from matplotlib import pyplot as plt

plt.figure()
plt.plot(range(len(v)), v)
plt.show()