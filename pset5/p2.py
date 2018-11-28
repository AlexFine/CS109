import random
import numpy as np
import matplotlib.pyplot as plt
import time

# Calculate x
def calc_x():
    x = 0
    for i in range(100):
        x += np.random.uniform(0,1)

    return x

# Simulate x
def sim_x():
    sim_n = 100000
    x_arr = []
    # Simulate
    for i in range(sim_n):
        x_arr.append(calc_x())
    # Generate percent array
    per_arr = perc(x_arr)
    # Print out results
    for i in range(100):
        print(i, (per_arr[i]/1000), "%")
        per_arr[i] = per_arr[i]/1000

    # Nums
    graph_arr = per_arr[30:60]
    temp = []
    for i in range(30):
        temp.append(i+30)
    plt.bar(temp, graph_arr)

    plt.draw()
    plt.pause(0.0001)
    time.sleep(1000000)


# Generate percent array
def perc(arr):
    arr_len = 100
    ret_arr = [0] * 100
    count = 0
    for i in arr:
        # print(i)
        ret_arr[int(round(i))] += 1
        # if (i > 47.5) and (i < 48.5):
        #     count += 1

    print(count)

    return ret_arr

print(sim_x())

"""
30 0.0 %
31 0.0 %
32 0.0 %
33 0.0 %
34 0.0 %
35 0.0 %
36 0.0 %
37 0.0 %
38 0.002 %
39 0.009 %
40 0.028 %
41 0.102 %
42 0.28 %
43 0.764 %
44 1.592 %
45 3.063 %
46 5.418 %
47 7.941 %
48 11.031 %
49 12.963 %
50 13.751 %
51 12.846 %
52 10.938 %
53 8.115 %
54 5.291 %
55 3.033 %
56 1.611 %
57 0.753 %
58 0.302 %
59 0.111 %
60 0.041 %
"""
