import random

# Bin
def calc_bin():
    successes = 0
    for i in range(1000):
        if random.uniform(0, 1) < 0.70:
            successes += 1
    # print(successes)
    return successes

#
def calc_answer():
    num = 0
    for i in range(100000):
        temp = calc_bin()
        if (temp > 690) and (temp < 710):
            num += 1

    print(num/10000)
    return 0

calc_answer()
