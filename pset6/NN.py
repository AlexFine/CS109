# Let's go

import random
import numpy as np
import math

# Logistic regression!!
LR = 0.0001
EPOCH_LEN = 10000
# DIMENTIONS
A = 50
B = 25

# FIX FOR NN
# master logic
def master():
    # load files
    filename = "netflix"
    path = "data/" + filename
    data_train, data_test = load_data(path)
    # Set x vs. y
    x_train, y_train = xy(data_train) # (nxm) (nx1)
    x_test, y_test = xy(data_test)
    # set m
    m = len([i for i in x_train[0]])
    # initialize weights
    weights = {
        "w1": rand_init(m, A), #mxA
        "w2": rand_init(A, B), #AxB
        "w3": rand_init(B, 1) #Bx1
    }
    # initialize bias
    bias = {
        "b1": [0] * A, #Ax1
        "b2": [0] * B, #Bx1
        "b3": 0 #1x1
    }
    # Train
    weights, bias = train(weights, bias, x_train, y_train, m, x_test, y_test)
    # Test
    """
    accuracy = test(w, b, x_test, y_test)
    # report accuracy
    return accuracy
    """

# FIX FOR NN
def test(w, b, x_test, y_test):
    correct = 0
    incorrect = 0
    # Test it out
    i = 0
    for row in x_test: # each row is mx1
        Z = np.dot(row, w)
        y_hat = sigmoid_v(Z)
        # Y_ht stuff
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

# FIX FOR NN
# train and optimize those weights!!
def train(old_w, old_b, x_train, y_train, m, x_test, y_test):
    # Set
    weights = old_w
    bias = old_b # don't mess w for now come back later
    x_train = np.asmatrix(x_train)
    y_train = np.asarray(y_train).reshape((-1,1))
    # train her
    for i in range(EPOCH_LEN):
        # First generate Z values
        # Set arrays outside of dic to ensure things are processed in correctly
        z1 = sigmoid_v(np.dot(x_train, weights["w1"]) + bias["b1"]) # (nxm)x(mxA) + Ax1 = nxA
        z2 = sigmoid_v(np.dot(z1, weights["w2"]) + bias["b2"]) # (nxA)x(AxB) + Bx1 = nxB
        z3 = sigmoid_v(np.dot(z2, weights["w3"]) + bias["b3"]) # (nxB )x(Bx1) + 1x1 = nx1
        # Create a dict
        Z = {
            "Z1": z1,
            "Z2": z2,
            "Z3": z3
        }
        # Now calculate all gradients
        # Partial of the log loss function of theta/partial of y hat
        pLT_pYH = y_train/Z["Z3"] - one_minus(y_train)/one_minus(Z["Z3"]) # nx1
        # yhat partial / theta 3 partial
        pYH_pT3 = np.dot(np.multiply(Z["Z3"], one_minus(Z["Z3"])).T, Z["Z2"]).T # (nx1)x(nx1) = (nx1)x(nxB) = Bxn
        # yhat partial / partial z2
        PYH_pZ2 = np.dot(np.multiply(Z["Z3"], one_minus(Z["Z3"])), weights["w3"].T) # nxB x 1xb = nxb
        # z2 partial / partial t2
        pZ2_pT2 = np.dot(np.multiply(Z["Z2"], one_minus(Z["Z2"])).T, Z["Z1"]).T # nxB.T x nxA = AxB
        # z2 partial / partial z1
        pZ2_pZ1 = np.dot(np.multiply(Z["Z2"], one_minus(Z["Z2"])), weights["w2"].T) # nxB AxB.T = nxA
        # z1 partial / theta 1 partial
        pZ1_pT1 = np.dot(np.multiply(Z["Z1"], one_minus(Z["Z1"])).T, x_train).T # nxA.T x nxm .T= mxA
        # Create a grad dictionary

        grads = {
            "grad_t3": np.dot(pYH_pT3, pLT_pYH.T), # (bxn)x(nx1) = bx1
            "grad_t2": mat_mul(pLT_pYH, PYH_pZ2, pZ2_pT2), # (nx1) (nxb) (AxB) = AxB
            "grad_t1": mat_mul(pLT_pYH, PYH_pZ2, pZ2_pZ1, pZ1_pT1),
            "grad_b1": calc_b(grads["grad_t1"], x_train),
            "grad_b2": calc_b(grads["grad_t2"], Z["Z1"]),
            "grad_b3": calc_b(grads["grad_t3"], Z["Z2"])
        }
        """
        # ok now update weights
        w1 = weights["w1"] + LR*grads["grad_t1"]
        w2 = weights["w2"] + LR*grads["grad_t2"]
        w3 = weights["w3"] + LR*grads["grad_t3"]
        b1 = bias["b1"] + LR*grads["grad_b1"]
        b2 = bias["b2"] + LR*grads["grad_b2"]
        b3 = bias["b3"] + LR*grads["grad_b3"]
        # Ok now append dict
        weights = {
            "w1": w1,
            "w2": w2,
            "w3": w3
        }
        # Biases
        bias = {
            "b1": b1,
            "b2": b2,
            "b3": b3
        }
        # # Set our Z layer
        # Z = np.dot(x_train, w.T) + b # x and w dot product
        # # Sigmoid yo
        # Z = sigmoid_v(Z) # nx1
        # # backprop to set gradients
        # temp = y_train - Z # nx1
        # b_grad = temp
        # grad = np.dot(x_train.T, temp.T) #nxm x 1xn = 1xm
        # # Scale by LR
        # grad = grad*LR
        # # Append
        # w = w.reshape((m,1))
        # w += grad
        # b += LR*(sum(grad)/len(grad))
        # w = w.reshape((m,))
        # print
        if i % 1000 == 0:
            print("Test Accuracy: ", test(w, b, x_test, y_test))
            print("Train Accuracy: ", test(w, b, x_train, y_train))
            print(i)
            """

    return weights, bias

# multiply sets of matrices together
def mat_mul(arr):
    # first multiply the first two arrays
    matrix = np.dot(arr[0], arr[1])
    # if there are more left
    count = 1
    while count < len(arr):
        matrix = np.dot(matrix, arr[count+1])
        count += 1

    return matrix

# Return one minus a matrix
def one_minus(matrix):
    return np.add(1, np.negative(matrix))

# custom function
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# define vectorized sigmoid
sigmoid_v = np.vectorize(sigmoid)

# Generate randomize values
def rand_init(m, A):
    w = []
    # loop
    for i in range(m):
        temp = []
        for j in range(A):
            temp.append(random.uniform(0, 1))
        w.append(temp)
    # return it yo
    return np.asmatrix(w).reshape(m, A)

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
