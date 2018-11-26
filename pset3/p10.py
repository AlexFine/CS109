str1 = "TTHHTHTTHTTTHTTTHTTTHTTHTHHTHHTHTHHTTTHHTHTHTTHTHHTTHTHHTHTTTHHTTHHTTHHHTHHTHTTHTHTTHHTHHHTTHTHTTTHHTTHTHTHTHTHTTHTHTHHHTTHTHTHHTHHHTHTHTTHTTHHTHTHTHTTHHTTHTHTTHHHTHTHTHTTHTTHHTTHTHHTHHHTTHHTHTTHTHTHTHTHTHTHHHTHTHTHTTHTHHTHTHTTHTTTHHTHTTTHTHHTHHHHTTTHHTHTHTHTHHHTTHHTHTTTHTHHTHTHTHHTHTTHTTHTHHTHTHTTT"
str2 = "HTHHHTHTTHHTTTTTTTTHHHTTTHHTTTTHHTTHHHTTHTHTTTTTTHTHTTTTHHHHTHTHTTHTTTHTTHTTTTHTHHTHHHHTTTTTHHHHTHHHTTTTHTHTTHHHHTHHHHHHHHTTHHTHHTHHHHHHHTTHTHTTTHHTTTTHTHHTTHTTHTHTHTTHHHHHTTHTTTHTHTHHTTTTHTTTTTHHTHTHHHHTTTTHTHHHHHHTHTHTHTHHHTHTTHHHTHHHHHHTHHHTHTTTHHHTTTHHTHTTHHTHHHTHTTHTTHTTTHHTHTHTTTTHTHTHTTHTHTHT"


def calc_rand():
    """
        STRATEGY:
        1. Break strings into arrays
        2. Iterate through each array
        3. Given a certain letter, the probability that the next letter is the same
        should be 50%, if they're evenly random. There should be no greater or lesser
        probability of the next letter occures given the original letter occured.
        4. Because each list is 100, we will simply count the times that the same letter
        is repeated.
        5. This would not prove that a certain string is random, but instead could prove
        that a certain string is not random.
        RESULTS:
        str1 is not random (38/100)
        str2 might be random (55/100)
    """
    # Raw variables
    str1PL = 0 # str1 letter counter
    str2PL = 0 # str2 letter counter
    # Create arrays
    arr1 = list(str1)
    arr2 = list(str2)
    # Iterate through arr1
    for i in range(99):
        if arr1[i] == arr1[i+1]:
            str1PL += 1
        if arr2[i] == arr2[i+1]:
            str2PL += 1
        else:
            pass

    print(str1PL)
    print(str2PL)

    return 0

calc_rand()
