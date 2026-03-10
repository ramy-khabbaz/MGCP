import numpy as np
from itertools import combinations, product

def burst_patterns(K, d):
    P = []
    for i in range(0, K - d + 1):
        P.append(list(range(i, i + d)))
    return P

def sortRowsByNonZerosAndL1Norm(inputMatrix):
    """
    Sort the rows of inputMatrix by:
      1) The number of non-zero elements (ascending)
      2) The L1 norm (ascending)
    """
    if inputMatrix.size == 0:
        return inputMatrix  # Return as is if empty
    
    # Number of nonzero entries in each row
    nonZeroCounts = np.count_nonzero(inputMatrix, axis=1)
    # L1 norm of each row
    l1Norms = np.sum(np.abs(inputMatrix), axis=1)
    
    # Sort by (nonZeroCounts, l1Norms). np.lexsort uses the last key first,
    # so we pass (l1Norms, nonZeroCounts) in that order.
    sort_idx = np.lexsort((l1Norms, nonZeroCounts))
    return inputMatrix[sort_idx, :]


def generateIntegerSolutions(n, k, minValue, maxValue, depth):
    """
    Generate all k-dimensional integer solutions x = (x1,...,xk) such that:
      1) Each xi is in [minValue, maxValue].
      2) sum(xi) == n.
      3) All xi are non-zero.
      4) sum(|xi|) - n <= 2*depth.
    """
    solutions = []
    for combo in product(range(minValue, maxValue+1), repeat=k):
        # sum(x_i) == n
        if sum(combo) != n:
            continue
        
        # all non-zero
        if any(x == 0 for x in combo):
            continue
        
        # L1 norm condition: (sum of abs(x_i)) - n <= 2*depth
        if (sum(abs(x) for x in combo) - n) <= 2*depth:
            solutions.append(combo)
    
    return np.array(solutions, dtype=int)


def patterns_beta(d, K, depth_val, len_last, c1):
    """
    Equivalent of patterns_beta in MATLAB. Builds patterns based on:
      - integer solutions that sum to abs(d) 
      - various combinations among K+c1 positions
      - and constraints like temp[K-1] <= len_last (1-based indexing from MATLAB)
    """
    P = []
    M = min(c1, abs(d) + 2*depth_val)
    minValue = -depth_val
    maxValue = abs(d) + depth_val

    for i in range(1, M+1):
        # Generate all integer solutions for the given i
        C = generateIntegerSolutions(abs(d), i, minValue, maxValue, depth_val)
        
        # Generate the nchoosek(1:K+c1, i) combinations.
        # In Python, we do zero-based indexing so we use range(K+c1).
        for comb_pos in combinations(range(K + c1), i):
            for row in C:
                temp = np.zeros(K + c1, dtype=int)
                # Place the values from 'row' into the positions 'comb_pos'
                for idx_val, val in zip(comb_pos, row):
                    temp[idx_val] = val
                
                # Corresponds to "if (temp(K) <= len_last)" in MATLAB,
                # adjusting for zero-based indexing => temp[K-1] 
                if temp[K-1] <= len_last:
                    P.append(temp)
    
    if len(P) == 0:
        return np.zeros((0, K + c1), dtype=int)
    
    P = np.array(P, dtype=int)
    # Sort by number of non-zeros and L1 norm
    P = sortRowsByNonZerosAndL1Norm(P)
    return P


def preCompute_Patterns(depth, K, len_last, lim, c1):
    """
    Precompute patterns for i=0..(lim-1),
    storing them in a list of length 'lim'.
    """
    P = [None] * lim
    for i in range(lim):
        #print(i)
        # Here, depth is assumed to be something indexable (like a list or array)
        P[i] = patterns_beta(i, K, depth[i], len_last, c1)
    return P

'''
P=preCompute_Patterns([1, 1, 0, 0, 0], 5, 7, 5, 4);
print(list(P))
print(P[2][4])

# Get the number of patterns for each i
num_patterns = sum([P[i].shape[0] if P[i] is not None else 0 for i in range(5)])
print(num_patterns)
print(P[0].shape[0])
'''