import numpy as np
import math
K = 0
L = 0

# master naive bayes classifier logic
def master():
    # First load in data
    file_name = "netflix"
    path = "data/" + file_name
    # Load data
    train_matrix, test_matrix = load_data(path)
    # Get the sets of data
    y0_set, y1_set = divide_data(train_matrix)
    # generate array aka model data
    y0_data, y1_data, y0_n, y1_n = gen_data(y0_set, y1_set)
    # Test accuracy
    accuracy = test_model(y0_data, y1_data, y0_n, y1_n, test_matrix)
    # Let us know how we did
    return accuracy

# Test the model
def test_model(y0_data, y1_data, y0_n, y1_n, test_matrix):
    correct = 0
    incorrect = 0
    # iterate through test matrix
    for row in test_matrix:
        # get input elements
        x = row[:-1]
        y = row[-1]
        # calculate the probability of each
        prob_0 = get_prob(y0_data, x, y0_n, y1_n)
        prob_1 = get_prob(y1_data, x, y1_n, y0_n)
        # pred
        pred = 0 if(prob_0 > prob_1) else 1
        # Append correct
        correct += 1 if (pred==y) else 0
        incorrect += 0 if (pred==y) else 1

    # Return
    return correct/(correct + incorrect)

# Get a prob
def get_prob(y_data, x, y1, y2):
    # print(y_data)
    """ using Naive Bayes we want:
        P(Y)*P(X_1|Y)*P(X_2|Y)*...*P(X_m|Y)
        using logs this is -->
        log(P(Y)) + sum log(P(X_i|Y))
    """
    prob = 0
    # generate p(Y)
    pY = y1/(y1+y2)
    lpY = math.log(pY)
    # append to prob
    prob += lpY
    # Iteratively generate P(X_i|Y)
    for i in range(len(x)):
        # Get probability
        temp_prob = y_data[i] if (x[i] == 1) else (1-y_data[i])
        # Take log
        l_prob = math.log(temp_prob) if (temp_prob != 0) else -1000000 # FIX ME
        # append to prob
        prob += l_prob

    # return aggregate probability
    return prob

# Generate data
def gen_data(y0_set, y1_set):
    # set m
    m = len([i for i in y0_set[0]])
    y0_n = 0
    y1_n = 0
    # set n's
    for row in y0_set:
        y0_n += 1
    y0_n += K # In case we gotta do this
    # set n's
    for row in y1_set:
        y1_n += 1
    y1_n += K # In case we gotta do this
    # Set up
    # these are the number of times that Xi equals 1
    # Set L in case we want to ... you know ... do MAP!!!
    y0_data = [L] * m
    y1_data = [L] * m
    """ We're going to divide these by n in order to get the probability that
    Xi = 1, given Y. We take 1- this value to find the probability that Xi = 0|Y
    """
    # Iterate through y0
    for row in y0_set:
        for i in range(len(row)):
            y0_data[i] += row[i]
    # Iterate through y1
    for row in y1_set:
        for i in range(len(row)):
            y1_data[i] += row[i]
    # Divide em out
    y0_data = [val/y0_n for val in y0_data]
    y1_data = [val/y1_n for val in y1_data]
    # return
    return y0_data, y1_data, y0_n, y1_n

# Divide data into y0 sets & y1_sets
def divide_data(matrix):
    # set em up
    y0_set = []
    y1_set = []
    # iterate yo
    for row in matrix:
        if row[-1] == 0:
            y0_set.append(row[:-1])
        else:
            y1_set.append(row[:-1])

    return y0_set, y1_set

# Load data from .txt files
def load_data(path):
    # set updated paths
    train_path = path + "-train.txt"
    test_path = path + "-test.txt"
    # load in training
    train_matrix = read_text(train_path)
    test_matrix = read_text(test_path)
    # Return it yo
    return train_matrix, test_matrix

# Just read text and return matrix
def read_text(path):
    # Load text file
    ret = []
    # Parse and load in text
    with open(path) as fileobj:
        m = next(fileobj)
        n = next(fileobj)
        for line in fileobj:
            line = line.split(' ')
            # print(line)
            for i in range(len(line)):
                line[i] = line[i].replace(":", "")
                line[i] = line[i].replace("\n", "")
                line[i] = int(line[i])
            #     print(ch)
            ret.append(line)

    return ret

print(master())
