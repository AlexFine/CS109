# Simulate 8
# int Recurse () {
# // equally likely to return 1, 2, or 3 int x = randomInteger (1 , 3);
# if (x == 1) return 3;
# else if (x == 2) return (5 + Recurse ()); else return (7 + Recurse ());
# }

from random import randint
import matplotlib.pyplot as plt

def recurse():
    x = randint(1,3)
    if (x == 1): return 3
    elif (x == 2): return (5 + recurse())
    elif (x == 3): return (7 + recurse())

def repeat():
    temp = []
    var_x = []
    for i in range(1000000):
        blip = recurse()
        var_x.append(blip*blip - 225)
        temp.append(blip)

    print(sum(var_x)/len(var_x))

    # Nums
    # graph_arr = per_arr[30:60]
    # temp = []
    # for i in range(30):
    #     temp.append(i+30)
    # plt.bar(temp, graph_arr)
    #
    # plt.draw()
    # plt.pause(0.0001)
    # time.sleep(1000000)


repeat()
