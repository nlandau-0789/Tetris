with open("random_samples", "r") as f:
    values = [sorted(int(i) for i in j.split()) for j in f.readlines()]

arr = [(i, j, k) for i in range(10) for j in range(i + 1, 10) for k in range(j + 1, 10)]
dic = {e: 0 for e in arr}

for e in values:
    dic[tuple(e)] += 1

arr2 = list(dic.values())
arr3 = range(len(arr2))
arr4 = [1000000 / len(arr2)]*len(arr2)

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(arr3, arr2)
plt.plot(arr3, arr4)

plt.ylim(0, 16000)
plt.show()
