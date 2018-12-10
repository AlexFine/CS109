# Let's go
import random
import numpy as np
import math

# Nearual Network!!
LR = 0.000001
EPOCH_LEN = 100000
# DIMENTIONS
A = 16
B = 4

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
    # set n
    n = 0
    for row in x_train:
        n += 1
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
    weights, bias = train(weights, bias, x_train, y_train, m, x_test, y_test, n)
    # Test
    accuracy = test(weights, bias, x_test, y_test)
    # report accuracy
    return accuracy

# Test NN accuracy
def test(w, b, x_test, y_test):
    correct = 0
    incorrect = 0
    loss = 0
    # Test it out
    i = 0
    print("weights: ", w["w1"])
    for row in x_test: # each row is mx1
        row = np.asarray(row)
        real = y_test[i]
        # hidden layer 1
        sig = np.dot(row, w["w1"]) # 1xm mxa = 1xa

        z = sigmoid_v(sig)
        # hidden layer 2
        sig2 = np.dot(z, w["w2"]) # 1xa axb = 1xb
        z2 = sigmoid_v(sig2)
        # hidden layer 3
        sig3 = np.dot(z2, w["w3"]) # 1xb bx1 = 1x1
        y_hat = sigmoid(sig3)
        # check if correct
        if abs(real - y_hat) <= 0.5:
            correct += 1
        else:
            incorrect += 1
        # append loss
        temp_loss = 0
        if real == 1:
            temp_loss = math.log(y_hat)
        else:
            temp_loss = math.log(1-y_hat)

        loss += temp_loss
        # iterate i
        i += 1

    print("Loss: ", loss)

    # Return final answer
    return correct/(correct + incorrect)

# train and optimize those weights!!
def train(old_w, old_b, x_train, y_train, m, x_test, y_test, n):
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
        print("DEBUGGIN: ", np.multiply(z1.T, z2).shape)
        # Create a dict
        Z = {
            "Z1": z1,
            "Z2": z2,
            "Z3": z3
        }
        # Now calculate all gradients
        # grad 3
        grad_t3 = np.dot(Z["Z2"].T, (y_train-Z["Z3"])) # (bxn) x (nx1) = bx1
        # grad 2
        grad_t2 = np.dot(weights["w3"], (y_train-Z["Z3"]).T) # (bx1)x(1xn) = bxn
        grad_t2 = np.dot(grad_t2, np.dot(Z["Z2"], one_minus(Z["Z2"]).T)) # (bxn)x(nxn) = bxn
        grad_t2 = np.dot(grad_t2, Z["Z1"]).T # (bxn)x(nxa) = bxa
        # grad 3
        grad_t1 = np.dot(weights["w3"], (y_train-Z["Z3"]).T) # (bx1)x(1xn) = bxn
        grad_t1 = np.dot(grad_t1, np.dot(Z["Z2"], one_minus(Z["Z2"].T))) # (bxn)x(nxn) = bxn
        temp = np.dot(np.dot(Z["Z1"], one_minus(Z["Z1"]).T), x_train) # nxn nxm = nxm
        grad_t1 = np.dot(grad_t1.T, weights["w2"].T) # nxb bxa = nxa
        grad_t1 = np.dot(grad_t1.T, temp).T # axn nxm = axm = mxa
        # Create a grad dictionary
        grads = {
            "grad_t3": LR*grad_t3,
            "grad_t2": LR*grad_t2,
            "grad_t1": LR*grad_t1,
        }
        # ok now update weights
        w1 = weights["w1"] + grads["grad_t1"]
        w2 = weights["w2"] + grads["grad_t2"]
        w3 = weights["w3"] + grads["grad_t3"]
        # print(w1)
        # Ok now append dict
        weights = {
            "w1": w1,
            "w2": w2,
            "w3": w3
        }

        if i % 10 == 0:
            print(test(weights, bias, x_train, y_train))
            print("Grads", grads["grad_t1"])
            # print(w3)
            # print(grads["grad_t3"])
            # print(i)

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
