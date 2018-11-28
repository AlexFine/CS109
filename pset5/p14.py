
N_SAMPLES = 10000

s() # returns a 1 with probability 0.5
e() # returns a 1 with probability 0.1

# Main Kahoona
def main_kahoona():
    # Generate sample data that's fake but statistically accurate
    # That's pretty cool
    arr = generate_a_lot_of_samples()
    # Generate sub array with criteria
    criteria = ["cold", "X2", "X7"] #
    sub_arr = get_sub(arr, criteria)
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
