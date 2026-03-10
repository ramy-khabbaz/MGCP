import random

def DNA_iid_channel(x, Pd, Pi, Ps):
    """
    Random edit channel - DNA iid case with shift tracking

    Parameters:
    x (list): Input DNA sequence as a list of characters.
    Pd (float): Probability of deletion.
    Pi (float): Probability of insertion.
    Ps (float): Probability of substitution.

    Returns:
    y (str): Output DNA sequence after applying errors.
    """
    y = ''         # Output DNA sequence
    nucleotides = 'ACGT'

    for i in range(len(x)):
        # Randomly sample the type of error based on probabilities
        r = random.choices([0, 1, 2, 3], weights=[1 - Pd - Pi - Ps, Pd, Pi, Ps])[0]

        if r == 0:  # No error
            y += x[i]

        elif r == 2:  # Insertion
            new_base = random.choice(nucleotides)
            y += new_base + x[i]

        elif r == 3:  # Substitution
            possible_bases = [base for base in nucleotides if base != x[i]]
            new_base = random.choice(possible_bases)
            y += new_base

    return y
