# b psudocode
def simulateBin(p = 0.4, n = 20):
    successes = 0
    for i in range(20):
        temp = random()
        if temp < p:
            successes += 1

    return successes

# c psudocode
def simulateGeo(p = 0.03):
    trials = 0
    success = False
    while !success:
        temp = random()
        trials += 1
        if temp < p:
            success = True

    return trials

# d psudocode
def simulateHypGeo(k = 5, p = 0.3):
    trials = 0
    successes = 0
    while successes < k:
        temp = random()
        trials += 1
        if temp < p:
            successes += 1

    return trials

# e psudocode
def simulatePoi(l = 3.1):
    n = 60000
    p = l/n
    successes = 0
    for i in range(n):
        temp = random()
        if temp < p:
            successes += 1

    return successes

# f psudocode 
def simulateExp(l = 3.1):
    n = 60000
    p = l/n
    success = False
    trials = 0
    while !success:
        temp = random()
        trials += 1
        if temp < p:
            success = True

    return trials
