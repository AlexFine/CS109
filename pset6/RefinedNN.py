# Let's go
import random
import numpy as np
import math

# Nearual Network!!
LR = 0.0001
EPOCH_LEN = 100000
# DIMENTIONS
A = 256
B = 128

# master logic
def master():
    # load files
    filename = "ancestry"
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
    model = {
        "w1": rand_init(m, A), #mxA
        "w2": rand_init(A, B), #AxB
        "w3": rand_init(B, 1), #Bx1
        "b1": [0] * A, #Ax1
        "b2": [0] * B, #Bx1
        "b3": 0 #1x1
    }
    # Train
    model = train(model, x_train, y_train, m, x_test, y_test, n)
    # Test
    accuracy = test(model, x_test, y_test)
    # report accuracy
    return accuracy

# POPULATE TEST FUNCTION
def test(model, x_test, y_test):
    correct = 0
    incorrect = 0
    loss = 0
    count = 0
    # Test it out
    for row in x_test: # each row is mx1
        # prop forward
        z1 = np.dot(row, model["w1"]) # 1xm mxa = 1xa
        z1 = tanh(z1 + model["b1"].T) # 1xa + 1xa = 1xa
        # z2
        z2 = np.dot(z1, model["w2"]) # 1xa axb = 1xb
        z2 = tanh(z2 + model["b2"].T) # 1xb + 1xb = 1xb
        # z3
        z3 = np.dot(z2, model["w3"]) # 1xb bx1 = 1x1
        z3 = sigmoid(z3 + model["b3"]) # 1
        # See if we're correct
        if (abs(y_test[count] - z3)) <= 0.5:
            correct += 1
        else:
            incorrect += 1
        # loss
        loss += math.log(z3) if y_test[count] == 1 else math.log(1-z3+0.00000001)
        # iterate count
        count += 1

    print("Loss: ", loss/count)
    # Return final answer
    return correct/(correct + incorrect)

# train and optimize those weights!!
def train(model, x_train, y_train, m, x_test, y_test, n):
    # Set
    ret_model = model # Return model
    x_train = np.asmatrix(x_train)
    y_train = np.asarray(y_train).reshape((-1,1))
    # train her
    for i in range(EPOCH_LEN):
        # Forward prop
        cache = forward_prop(ret_model, x_train, y_train)
        # Backprop
        grads = backprop(cache, ret_model, x_train, y_train, n)
        # Update weights
        ret_model = update_weights(grads, ret_model)
        # check out how we're doing
        if i % 100 == 0:
            print("Train Accuracy: ", test(ret_model, x_train, y_train))
            # print("Test Accuracy: ", test(ret_model, x_test, y_test))
            pass

    return ret_model

# update those weights
def update_weights(grads, model):
    ret_model = model
    # update
    ret_model["w1"] = model["w1"] - LR*grads["w1"]
    ret_model["w2"] = model["w2"] - LR*grads["w2"].T
    ret_model["w3"] = model["w3"] - LR*grads["w3"]
    ret_model["b1"] = model["b1"] - LR*grads["b1"]
    ret_model["b2"] = model["b2"] - LR*grads["b2"]
    ret_model["b3"] = model["b3"] - LR*grads["b3"]
    # return new model
    return ret_model

# Back prop
def backprop(cache, model, x_train, y_train, n):
    # Return gradients
    grads = {}
    # Main sigmoid loss
    ll = cache["z3"] - y_train # nx1 - nx1 = nx1
    grad_w3 = (1/n)*np.dot(ll.T, cache["z2"]).T # (1xn) (nxb) = 1xb.T = bx1
    # Combined dz2 grad
    dz2 = np.dot(ll, model["w3"].T) # (nx1) (1xb) = nxb
    dz2 = np.multiply(dz2, tan_prime(cache["z2"])) # nxb nxb = nxb
    grad_w2 = (1/n)*np.dot(cache["z1"].T, dz2).T # bxn nxa = axb
    # Fianl weight gradient
    dz1 = np.dot(dz2, model["w2"].T) # nxb bxa = nxa
    dz1 = np.multiply(dz1, tan_prime(cache["z1"])) # nxa nxa = nxa
    grad_w1 = (1/n)*np.dot(x_train.T, dz1) # mxn nxa = mxa
    # now get those b gradients
    grad_b3 = (1/n)*np.sum(ll) # 1x1
    grad_b2 = (1/n)*np.sum(dz2) # bx1
    grad_b1 = (1/n)*np.sum(dz1) # ax1
    # append to grads
    grads["w1"], grads["w2"], grads["w3"], grads["b1"], grads["b2"], grads["b3"] = grad_w1, grad_w2, grad_w3, grad_b1, grad_b2, grad_b3
    # return
    return grads

# Forward prop
def forward_prop(model, x_train, y_train):
    # Set return cache
    cache = {}
    # Forward prop through z1
    z1 = np.dot(x_train, model["w1"]) # (nxm) (mxa) = nxa
    z1 = tanh(z1 + model["b1"]) # nxa + ax1 = nxa
    # Onto z2
    z2 = np.dot(z1, model["w2"]) # (nxa) (axb) = nxb
    z2 = tanh(z2 + model["b2"]) # nxb + bx1
    # Finally y hat
    z3 = np.dot(z2, model["w3"]) # (nxb) (bx1) = nx1
    z3 = sigmoid_v(z3 + model["b3"]) # nx1 + 1x1 = nx1
    # Append cache
    cache["z1"], cache["z2"], cache["z3"] = z1, z2, z3
    # return cache
    return cache

# derivative of tanh x
def tan_prime(arr):
    e1 = np.power(math.e, -arr)
    e2 = np.power(math.e, -2*arr) + 1
    ret = 2*np.divide(e1, e2)
    ret = np.power(ret, 2)
    return ret

# Set tanh function
def tanh(arr):
    return np.tanh(arr)

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
