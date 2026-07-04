import matplotlib.pyplot as plt

l = ["3-1024", "1.41-1024", "5-1024", "5-1024-2", "3-1024-2", "0-8192", "3-256"]

for n in l:
    with open("logs/" + n, "r") as f:
        nums = [float(i.strip()) for i in f.readlines()]
        t = [i for i in range(len(nums))]
        plt.plot(t, nums, label = n)

plt.legend()
plt.show()