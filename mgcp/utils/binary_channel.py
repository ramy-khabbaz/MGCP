import numpy as np

def binary_channel(x, Pd, Pi, Ps):
    """
    Random edit channel - binary case.

    Parameters:
        x (list): Input binary sequence.
        Pd (float): Probability of deletion.
        Pi (float): Probability of insertion.
        Ps (float): Probability of substitution.

    Returns:
        tuple: y (message with errors)
    """
    s = 0
    errs = []
    y = []

    for i in range(len(x)):
        r = np.random.choice([0, 1, 2, 3], p=[1 - Pd - Pi - Ps, Pd, Pi, Ps])
        if r == 0:
            y.append(x[i])
        elif r == 1:  # Deletion
            s -= 1
            errs.append([-1, i])
        elif r == 2:  # Insertion
            y.append(np.random.randint(0, 2))  # Random binary value
            y.append(x[i])
            errs.append([1, i])
            s += 1
        elif r == 3:  # Substitution
            y.append(1 - x[i])  # Flip the bit
            errs.append([0, i])

    return y
