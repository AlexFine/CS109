#
# int roll() {
# int total = 0;
# while
# //
# int
# ( true ) equally roll =
# {
# likely to return randomInteger (1 ,
# 1 ,... ,6 6);
# total += roll ;
# // exit condition
# if (roll >= 3) break;
# }
# return total ; }
from random import randint

def roll():
    total = 0
    count = 0
    while (True):
        count += 1
        rand = randint(1,6)
        total += rand
        if rand >= 3:
            break

    return count

def simulate():
    total_list = []
    for i in range(100000):
        total_list.append(roll())

    print(sum(total_list)/len(total_list))

simulate()
