
# Use a beta distribution to construct our probability
def probFlu():
    # Set initial beta values
    a = 1
    b = 1
    # Set the criteria we're selecting for
    criteria = {
    "Flu": None,
    "Exposure": 1,
    "X2": 1
    }
    # Append beta a small number of times, we won't need a lot
    for i in range(15):
        a, b = append_Beta(a, b, criteria)
    # Find expected of the beta
    prob = a/(a+b)
    # return
    return prob

# Create a sample, and then append Beta
def append_Beta(a, b, real_criteria):
    # Create a new criteria dictionary,
    # only comparing the criterias but not the values for those criteria
    pseudo_criteria = dict.fromkeys(real_criteria)
    # Use ternary operators to update our belief about stress and exposure
    s = pseudo_criteria["Stress"] if ("Stress" in pseudo_criteria) else s
    e = pseudo_criteria["Exposure"] if ("Exposure" in pseudo_criteria) else e
    # Create a prior flu & prior cold boolean, this is NOT a probability 
    prior_flu = probFlu(s, e)
    prior_cold = priorCold(s, e)
    # Given the prior flu boolean, check prob of flu given symptoms
    for symptom in real_criteria:
        pseudo_criteria[symptom] = probSymptom(symptom, prior_flu, prior_cold)
    # Check to see if pseudo_criteria matches real_criteria
    # if so append our a and b
    a += 1 if (pseudo_criteria=real_criteria) else 0
    b += 0 if (pseudo_criteria=real_criteria) else 1
    # return updated values
    return a, b
