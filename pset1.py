# PSet one doc to check work
from random import randint
#randint(1, 100)
# def p8():
#     temp_vec = []
#     for a in range(6):
#         for b in range(6):
#             for c in range(6):
#                 for d in range(6):
#                     for e in range(6):
#                         for f in range(6):
#                             temp_vec.append(a)
#                             temp_vec.append(b)
#                             temp_vec.append(c)
#                             temp_vec.append(d)
#                             temp_vec.append(e)
#                             temp_vec.append(f)
#                             if (a == b) or
#

def p15():
    # Sum starts at 0
    num_sum = 0
    # Set random
    rand1 = 0
    rand2 = 0
    # Win counter
    P1_wins = 0
    P2_wins = 0
    # Repeat process 100000 times
    for i in range(1000000):
        while num_sum <= 100:
            rand1 = randint(1, 100)
            num_sum += rand1
        while (num_sum <= 200) and (num_sum > 100):
            rand2 = randint(1, 100)
            num_sum += rand2
        # Player one wins
        if rand1 > rand2:
            P1_wins += 1
        # player two wins
        elif(rand2 > rand1):
            P2_wins += 1
        # They got the same number, tie
        else:
            pass
        num_sum = 0

    print("P1 Wins: ", P1_wins)
    print("P2 Wins: ", P2_wins)
    print("Ties: ", 1000000 - P1_wins - P2_wins)

p15()
