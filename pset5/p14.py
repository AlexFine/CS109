# Number of samples
N_SAMPLES = 10000
# Define functions we use later
s() # returns a 1 with probability 0.5
e() # returns a 1 with probability 0.1
gen_sub_arr(arr, criteria) # Generates a sub array of arr, given the criteria

# Main Kahoona
def probFlu():
    """ Creates samples, lists the specific criteria to test for,
    which in this case is Exposure and X2, and finds the probability
    of the flu given these criteria """
    # Generate sample data that's fake but statistically accurate
    # That's pretty cool
    arr = generate_a_lot_of_samples()
    # Generate sub array with criteria
    criteria = ["Exposure", "X2"]
    # All occurances of data that fit the criteria
    arr_with_criteria = gen_sub_arr(arr, criteria)
    # Everyone who fits the criteria, and has the flu
    arr_with_flu = gen_sub_arr(arr_with_criteria, ["Flu"])
    # Find the proportion of those with the flu, given that they have the criteria
    prob = len(arr_with_flu)/len(arr_with_criteria)
    # Return
    return prob

# Generate lots of samples
def generate_a_lot_of_samples():
    arr = []
    # Generate a lot of samples
    for i in range(N_SAMPLES):
        arr.append(generate_sample)
    # Return arr
    return arr

# Generate a sample
def generate_sample():
    # Create a sample dictionary to ensure data is set correctly
    sample = {}
    sample["cold"] = Ber(probCold(s(),e()))
    sample["flu"] = Ber(probFlu(s(),e()))
    # Iterate through for each Xi
    for i in range(10):
        sample["X" + str(i+1)] = probSymptom((i+1), f, c)

    return sample
