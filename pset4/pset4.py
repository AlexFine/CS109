import numpy as np
import csv
import scipy.stats
import math

# Num 10
def max():
    n = 3
    max_num = -1
    arr = np.random.randint(0, high=1000000000000000, size=n)# set array
    count = 0
    for i in range(n):
        if arr[i] > max_num:
            count += 1
            max_num = arr[i]

    return count

def test():
    avg = 0
    n = 10000
    for i in range(n):
        avg += max()
        if (i%10 == 1):
            print(avg/i)

    print("Average: ", avg/n)

# Num 12
# Load data into matrix
# Find number of items in column with attribute x
# Reads a files into a 2d array. There are
# other ways of doing this (do check out)
# numpy. But this shows
def loadCsvData(fileName):
    matrix = []
    # open a file
    with open(fileName) as f:
        reader = csv.reader(f)
        next(reader)
        # loop over each row in the file
        for row in reader:
            # cast each value to a float
            """Survived,Pclass,Name,Sex,Age,Siblings/Spouses Aboard,Parents/Children Aboard,Fare
            0,3,Mr. Owen Harris Braund,male,22,1,0,7.25
            """
            doubleRow = []
            for value in row:
                doubleRow.append(str(value))

            # store the row into our matrix
            matrix.append(doubleRow)
    return matrix


def loadCsvDataYay(fileName):
    matrix = []
    # open a file
    with open(fileName) as f:
        reader = csv.reader(f)
        next(reader)
        # loop over each row in the file
        for row in reader:
            # cast each value to a float
            """Survived,Pclass,Name,Sex,Age,Siblings/Spouses Aboard,Parents/Children Aboard,Fare
            0,3,Mr. Owen Harris Braund,male,22,1,0,7.25
            """
            doubleRow = []
            doubleRow.append(float(row[0]))
            doubleRow.append(str(row[1]))
            # for value in row:
            #     doubleRow.append(str(value))

            # store the row into our matrix
            matrix.append(doubleRow)
    return matrix

# Number 13
def fuckyesyesysyysys():
    personA = loadCsvDataYay('personKeyTimingA.csv')
    personB = loadCsvDataYay('personKeyTimingB.csv')
    email = loadCsvDataYay('email.csv')
    a_count = 1
    b_count = 1
    count = 0
    prev_a = 0
    prev_b = 0

    # for row in personA:
    #     print(row[0] - prev_a)
    #     print(prev_a)
    #     # a_count += (row[0] - prev_a)
    #     a_count += (row[0] - prev_a)*(row[0] - prev_a)
    #     count += 1
    #     prev_a = row[0]
    #
    # for row in personB:
    #     # b_count += (row[0] - prev_b)
    #     b_count += (row[0] - prev_b)*(row[0] - prev_b)
    #     count += 1
    #     prev_b = row[0]
    # scipy.stats.norm(0, 1).pdf(0)
    # math.log


    prev = 0
    for row in email:
        a_count *= scipy.stats.norm(7.411, 51.549).pdf(row[0] - prev)
        b_count *= scipy.stats.norm(8.036, 60.32).pdf(row[0] - prev)
        prev = row[0]
    #
    print("a_count", a_count)
    print("b_count", b_count)

# calculate answers
def loops():
    # Set variables
    total = 0
    total_survived = 0
    # Load matrix
    matrix = loadCsvData('titanic.csv')
    # Loop through each row in matrix
    for row in matrix:
        # if (float(row[4]) <= 10) and (row[1] == "3"):
        if (row[1] == "3"):
            total += 1
            total_survived += float(row[-1])
            # if(row[0] == "1"):
            #     total_survived += 1

    print("Prob: ", total_survived/total)
    print(total_survived)
    print(total)


fuckyesyesysyysys()
