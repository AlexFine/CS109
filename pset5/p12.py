import csv
import random
import statistics

N = 100000
# Reads a files into a 2d array. There are
# other ways of doing this (do check out)
# numpy. But this shows
def loadCsvData(fileName):
	arr = []
	# open a file
	with open(fileName) as f:
		reader = csv.reader(f)
		# loop over each row in the file
		for row in reader:
			# cast each value to a float
			for value in row:
				arr.append(int(value))
			# store the row into our matrix
	return arr

# Master function
def master():
    data_arr = loadCsvData('peerGrades.csv')
    # Part a
    print(sample_mean(data_arr))
    # Part b
    print(var_w_5(data_arr))
    # Part c
    print(var_w_5_median(data_arr))
    # Part d
    print(expected_of_both(data_arr))
# Expected value of BOTH YO
def expected_of_both(arr):
    loop_len = 5
    blip = []
    var_arr = []
    # Simulate the bitch
    for i in range(N):
        temp = []
        for i in range(loop_len):
            temp.append(random.choice(arr))
        # calc avg.
        median = statistics.median(temp)
        blip.append(median)

    blop = []
    for i in range(N):
        temp = []
        for i in range(loop_len):
            temp.append(random.choice(arr))
        # calc avg.
        avg = sum(temp)/len(temp)
        blop.append(avg)

    return sum(blip)/len(blip), sum(blop)/len(blop)

# Bootstrap to create an array of 5 averages
def var_w_5_median(arr):
    loop_len = 5
    e_2_arr = []
    var_arr = []
    # Simulate
    for i in range(N):
        temp = []
        for i in range(loop_len):
            temp.append(random.choice(arr))
        # calc avg.
        median = statistics.median(temp)
        e_2_arr.append(median*median)
    # Print variance
    e2_sum = sum(e_2_arr)/len(e_2_arr)
    samp, _ = expected_of_both(arr)
    # Variance
    var = e2_sum - samp*samp
    # return
    return var

# Bootstrap to create an array of 5 averages
def var_w_5(arr):
    loop_len = 5
    e_2_arr = []
    var_arr = []
    # Simulate
    for i in range(N):
        temp = []
        for i in range(loop_len):
            temp.append(random.choice(arr))
        # calc avg.
        avg = sum(temp)/len(temp)
        e_2_arr.append(avg*avg)
    # Print variance
    e2_sum = sum(e_2_arr)/len(e_2_arr)
    samp = sample_mean(arr)
    # Variance
    var = e2_sum - samp*samp
    # return
    return var


# Calculate the sample mean of the arra
def sample_mean(arr):
    arr_sum = 0
    # Calculate sum
    for i in range(len(arr)):
        arr_sum += arr[i]
    # Set return
    ret = arr_sum/len(arr)

    return ret

master()
