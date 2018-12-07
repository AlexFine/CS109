import random
import numpy as np
import math

# Logistic regression!!
LR = 0.001
EPOCH_LEN = 10000

# master logic
def master():
    # load files
    filename = "simple"
    path = "data/" + filename
    data_train, data_test = load_data(path)
    # Set x vs. y
    x_train, y_train = xy(data_train) # (nxm) (nx1)
    x_test, y_test = xy(data_test)
    # set m
    m = len([i for i in x_train[0]])
    # initialize weights
    w = rand_init(m) # (mx1)
    b = 0 # (1x1)
    # Train
    w, b = train(w, b, x_train, y_train, m)
    # Test
    accuracy = test(w, b, x_test, y_test)
    # report accuracy
    return accuracy

def test(w, b, x_test, y_test):
    correct = 0
    incorrect = 0
    # Test it out
    i = 0
    for row in x_test: # each row is mx1
        Z = np.dot(row, w)
        y_hat = sigmoid_v(Z)
        print(y_hat)
        print(y_test[i])
        if (y_test[i] == 1) and (y_hat > 0.5):
            correct += 1
        elif(y_test[i] == 0) and (y_hat <= 0.5):
            correct += 1
        else:
            incorrect += 1
        # iterate i
        i += 1

    # Return final answer
    return correct/(correct + incorrect)

# train and optimize those weights!!
def train(old_w, old_b, x_train, y_train, m):
    # Set
    w = np.asarray(old_w)
    b = old_b # don't mess w for now come back later
    x_train = np.asmatrix(x_train)
    # train her
    for i in range(EPOCH_LEN):
        # Set our Z layer
        Z = np.dot(x_train, w.T) + b # x and w dot product
        # Sigmoid yo
        Z = sigmoid_v(Z) # nx1
        # backprop to set gradients
        temp = y_train - Z # nx1
        b_grad = temp
        grad = np.dot(x_train.T, temp.T) #nxm x 1xn = 1xm
        # Scale by LR
        grad = grad*LR
        # Append
        w = w.reshape((m,1))
        w += grad
        b += LR*(sum(grad)/len(grad))
        w = w.reshape((m,))
        # print
        if i % 1000 == 0:
            print(i)


    return w, b

# custom function
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# define vectorized sigmoid
sigmoid_v = np.vectorize(sigmoid)

# Generate randomize values
def rand_init(m):
    w = []
    # loop
    for i in range(m):
        w.append(random.uniform(0, 1))
    # return it yo
    return w

# Break into x y data yo
def xy(matrix):
    # set
    x = []
    y = []
    # loop an append
    for row in matrix:
        # set
        x_temp = row[:-1]
        y_temp = row[-1]
        # append
        x.append(x_temp)
        y.append(y_temp)
    # return
    return x, y

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
