import random

def DNA_iid_channel(x, Pd, Pi, Ps):
    y = ''
    nucleotides = 'ACGT'
    errors = 0
    for i in range(len(x)):
        r = random.choices([0, 1, 2, 3], weights=[1 - Pd - Pi - Ps, Pd, Pi, Ps])[0]
        if r == 0:    # no error
            y += x[i]
        elif r == 1:  # deletion
            errors += 1
            pass      # base dropped, nothing appended
        elif r == 2:  # insertion
            errors += 1
            new_base = random.choice(nucleotides)
            y += new_base + x[i]
        elif r == 3:  # substitution
            errors += 1
            possible_bases = [base for base in nucleotides if base != x[i]]
            new_base = random.choice(possible_bases)
            y += new_base
    return y, errors
