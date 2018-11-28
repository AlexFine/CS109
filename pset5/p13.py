# PSEODOCODE
def administer():
    # Set the initial beta variables to 1
    d1_a = 1
    d1_b = 1
    d2_a = 1
    d2_b = 1
    # Repeat drug administering logic 100 times
    for i in range(100):
        # Retrieve the initial sampled values
        samp_1 = sampleBeta(d1_a, d1_b)
        samp_2 = sampleBeta(d2_a, d2_b)
        # Decide which drug is administered
        if (samp_1 > samp_2):
            R = giveDrug("one")
            # Append beta values accordingly
            d1_a += 1 if(R) else 0
            d1_b += 0 if(R) else 1
        else:
            R = giveDrug("one")
            # Append beta values accordingly
            d2_a += 1 if(R) else 0
            d2_b += 0 if(R) else 1

    return d1_a, d1_b, d2_a, d2_b
