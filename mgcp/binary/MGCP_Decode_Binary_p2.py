import numpy as np
from reedsolo import RSCodec
import reedsolo
import math
from .CalculateProbasBinary import CalculateProbas

def MGCP_Decode_Binary_p2(y, n, l, N, K, c1, c2, q, maxSize, marker_period, P0, Pd, Pi, Ps, codebook, dmin):
    # Initializations
    uhat = None
    delta = len(y) - n
    size_sub = 0
    final_sub_deletion_patterns1 = []
    d = []
    errs = 0
    markerSize = 3
    num_blocks = K + c1
    num_markers = math.ceil(num_blocks // 2)

    # RS encoder setup
    rsEncoder = RSCodec((c1 + c2), c_exp=l)
    
    # RS decoder setup
    rsDecoder = RSCodec((c1 + c2), c_exp=l)

    # Fast check
    if delta == 0:
        # Decode check parities
        seg = y[-len(codebook[0]):]
        Par, dist = find_min_levenshtein_row(codebook, seg, dmin)
        Par=[Par]

        size_sub += 1
        yE = y[:((num_blocks * l) + (num_markers * markerSize))]
        lengths = np.ones(K + c1, dtype=int) * l

        # Update lengths array to add marker size after every 2 blocks
        for i in range(1, K + c1, 2):  # Python indexing starts at 0, so adjust the range
            lengths[i] += markerSize

        Y = divide_vector(yE, lengths, 3)
        if Y:
            Y.extend(Par)
            erasure_pattern = [0] * (N - c2) + [1] * (c2)
            ones_positions = [i for i, value in enumerate(erasure_pattern) if value == 1]

            try:
                Uhat = list(rsDecoder.decode(Y,erase_pos=ones_positions)[0])
            except reedsolo.ReedSolomonError as e:
                Uhat = None

            if Uhat:
                Xhat = list(rsEncoder.encode(Uhat))
                non_erasure_idx = [i for i, val in enumerate(erasure_pattern) if val != 1]
                mismatches = sum(Xhat[i] != Y[i] for i in non_erasure_idx)
                if mismatches > (c1 + c2) // 2:
                    errs = -1
                if Xhat[K + c1 : K + c1 + c2] == Par and errs != -1:
                    uhat_blocks = decimal_to_binary_blocks(Uhat, l)
                    uhat = [int(bit) for block in uhat_blocks for bit in block]
                    return uhat
    
    # Trellis Calculation
    
    d = deletion_error_location(y, (K+c1)//marker_period, delta, l*marker_period, 3, c1//marker_period, P0, Pd, Pi, Ps)
    
    # Decode check parities
    seg = y[-len(codebook[0])+sum(d)-delta:]
    Par, dist = find_min_levenshtein_row(codebook, seg, dmin)
    Par=[Par]

    delta = sum(d)
    yE = y[:(num_blocks * l) + (num_markers * markerSize) + delta]

    d = arrange_pattern(d)
    d1 = d

    # Phase 1: Quick Check
    lengths = np.ones(num_blocks, dtype=int) * l 
    for i in range(1, K + c1, 2): 
        lengths[i] += markerSize

    lengths = lengths + d
    Y = divide_vector(yE, lengths, 3)

    size_sub += 1
    if Y:
        Y.extend(Par)
        erasure_pattern = [0] * (N - c2) + [1] * (c2)
        for i in range(len(d)):
            if d[i] != 0:
                erasure_pattern[i] = 1

        ones_positions = [i for i, value in enumerate(erasure_pattern) if value == 1]

        if np.count_nonzero(d) <= c1:
            Y = [0 if value > q - 1 else value for value in Y]
            try:
                Uhat = list(rsDecoder.decode(Y,erase_pos=ones_positions)[0])
            except reedsolo.ReedSolomonError as e:
                Uhat = None

            if Uhat:                
                Xhat = list(rsEncoder.encode(Uhat))

                non_erasure_idx = [i for i, val in enumerate(erasure_pattern) if val != 1]
                mismatches = sum(Xhat[i] != Y[i] for i in non_erasure_idx)
                if mismatches > c1 // 2:
                    errs = -1

                if Xhat[K + c1 : K + c1 + c2] == Par and errs != -1:
                    uhat_blocks = decimal_to_binary_blocks(Uhat, l)
                    uhat = [int(bit) for block in uhat_blocks for bit in block]
                    return uhat

    # Phase 2: Local Apportionment

    consecutive_sequences = []
    current_sequence = []

    # Find consecutive sequences in d
    for i in range(len(d)):
        if d[i] != 0:
            current_sequence.append(i)
        elif current_sequence:
            consecutive_sequences.append(current_sequence)
            current_sequence = []
    if current_sequence:
        consecutive_sequences.append(current_sequence)

    # Create partial sub-deletion patterns for each consecutive sequence
    partial_sub_deletion_patterns = []
    for current_sequence in consecutive_sequences:
        partial_sub_deletion_pattern = np.zeros(len(d))
        partial_sub_deletion_pattern[current_sequence] = np.take(d, current_sequence)
        partial_sub_deletion_patterns.append(partial_sub_deletion_pattern)

    # Expand each partial sub-deletion pattern
    expanded_partial_sub_deletion_patterns1 = []
    for partial_sub_deletion_pattern in partial_sub_deletion_patterns:
        expanded_partial_sub_deletion_pattern = expand_deletion_pattern1(partial_sub_deletion_pattern, K + c1)
        expanded_partial_sub_deletion_patterns1.append(expanded_partial_sub_deletion_pattern)

    # Generate sub-deletion patterns for each expanded partial sub-deletion pattern
    all_sub_deletion_patterns1 = []
    for idx, expanded_partial_sub_deletion_pattern in enumerate(expanded_partial_sub_deletion_patterns1):
        if np.all(sum(expanded_partial_sub_deletion_pattern) == 0):  # Skip empty expanded patterns
            continue
        sub_deletion_patterns = generate_sub_deletion_patterns1(
            expanded_partial_sub_deletion_pattern,
            partial_sub_deletion_patterns[idx]
        )
        all_sub_deletion_patterns1.append(sub_deletion_patterns)

    # Generate all combinations of sub-deletion patterns from different cells
    final_sub_deletion_patterns1 = generate_combinations(all_sub_deletion_patterns1, l, maxSize*10)
    size_sub += len(final_sub_deletion_patterns1)

    # Check all patterns generated after local apportionment
    for pattern_idx in range(min(len(final_sub_deletion_patterns1), maxSize)):
        d2 = final_sub_deletion_patterns1[pattern_idx].astype(int).tolist()
        if not d2:
            continue
        uhat, valid = check_pattern(yE, d2, K, c1, c2, l, q, Par, rsDecoder, rsEncoder, N,num_blocks,markerSize)
        if valid:
            return uhat                  
                           
    return uhat

def binary_to_decimal_blocks(binary_message, block_length):

    # Ensure the binary message is divisible by block_length
    if len(binary_message) % block_length != 0:
        raise ValueError("The length of the binary message must be divisible by the block length.")

    # Group the binary message into blocks of block_length
    blocks = [binary_message[i:i + block_length] for i in range(0, len(binary_message), block_length)]

    # Convert each binary block to decimal
    decimal_blocks = [int("".join(map(str, block)), 2) for block in blocks]

    return decimal_blocks

# Helper function to decode with repetitions
def rep_decode_gamma(y, repetitions, k):
    uhat = []
    while len(uhat) < k and len(y) > 0:
        t = min(repetitions, len(y))
        seg = y[:t]
        bit = np.bincount(seg).argmax()  # Mode of the segment
        uhat.append(bit)
        y = y[t:]
    
    if len(uhat) < k:
        uhat.extend(np.random.randint(0, 2, size=(k - len(uhat),)).tolist())
    elif len(uhat) > k:
        uhat = uhat[:k]
    
    return uhat

# Divide vector for binary sequences
def divide_vector(x, y, marker_size):

    Y = []  # Final decimal result
    n = len(y)  # Number of parts

    parts = [None] * n  # Initialize a list to store binary parts
    start_idx = 0  # Initialize starting index

    for i in range(n):
        # Calculate ending index for the current part
        end_idx = start_idx + y[i]

        # Remove marker only after every 2 blocks
        if (i + 1) % 2 == 0:
            part = x[start_idx:end_idx - marker_size]
        else:
            part = x[start_idx:end_idx]

        if not part:  # If the part is empty
            Y.append(0)
        else:
            # Convert binary sequence to decimal
            decimal_value = int("".join(map(str, part)), 2)
            Y.append(decimal_value)

        parts[i] = part
        start_idx = end_idx  # Update starting index for the next part

    return Y

# Function to locate deletion error
def deletion_error_location(r, v, delta, block_size, l, c1, P0, Pd, Pi, Ps):

    # v += c1

    max_delta = abs(delta) + 2
    max_shift_per_block = max_delta

    P = np.zeros((v + 1, 2 * max_delta + 1))
    Q = np.zeros((v + 1, 2 * max_delta + 1))
    P[0, max_delta] = 1  # Initial state with zero shift

    # Recalculate L matrix for current error probabilities
    L = np.zeros((8, 2*max_delta + 1))
    for m_prime in range(8):
        for shift in range(-max_delta, max_delta + 1):
            L[m_prime, shift + max_delta] = CalculateProbas(m_prime, shift, block_size, P0, Pd, Pi, Ps)

    # Phase 1: Compute likelihoods and record preceding states
    for i in range(1, v + 1):
        for omega in range(-max_delta, max_delta + 1):

            current_m = decode_rib(r, i - 1, omega, block_size, l)

            if current_m == -1:
                continue

            m_idx = int(''.join(map(str, current_m)), 2)

            max_val = 0.0
            best_mu = 0
            for mu_prime in range(-max_shift_per_block, max_shift_per_block + 1):
                prev_omega = omega - mu_prime
                if not (-max_delta <= prev_omega <= max_delta):
                    continue

                likelihood = P[i - 1, prev_omega + max_delta] * L[m_idx, mu_prime + max_shift_per_block]
                if likelihood > max_val:
                    max_val = likelihood
                    best_mu = mu_prime

            P[i, omega + max_delta] = max_val
            Q[i, omega + max_delta] = best_mu

        # Normalize probabilities
        C = np.sum(P[i, :])
        if C > 0:
            P[i, :] /= C

    # Phase 2: Trace the optimal path
    Z = np.zeros(v + 1, dtype=int)
    Z[-1] = np.argmax(P[-1, :]) - max_delta
    #Z[-1] = delta

    d = np.zeros(v, dtype=int)
    for i in range(v, 0, -1):
        omega = Z[i]
        shift = Q[i, omega + max_delta]
        Z[i - 1] = omega - int(shift)
        d[i - 1] = int(shift)

    return d.tolist()

# Helper function to decode rib
def decode_rib(r, i, omega, block_size, marker_size):

    block_with_marker_size = block_size + marker_size
    start_block_index = i * block_with_marker_size + omega
    marker_start_index = start_block_index + block_size
    marker_end_index = marker_start_index + marker_size

    if 0 <= marker_start_index < len(r) and marker_end_index <= len(r):
        return r[marker_start_index:marker_end_index]
    return -1

def expand_deletion_pattern1(d, v):
    expanded_d = np.zeros(v, dtype=int)
    for i, val in enumerate(d):
        if val != 0:
            di = abs(val)
            expansion = math.ceil(di / 2)  # Calculate ceil(di / 2)
            start = int(max(0, i - expansion))  # Adjust the start using expansion
            end = int(min(v, i + expansion + 1))
            expanded_d[start:end] = 1
    return expanded_d

def generate_sub_deletion_patterns1(expanded_pattern, partial_pattern):
    v = len(expanded_pattern)
    cluster_patterns = []

    # Identify clusters in the expanded pattern
    cluster_start = []
    cluster_end = []
    i = 0
    while i < v:
        if expanded_pattern[i] > 0:
            cluster_start.append(i)
            while i < v and expanded_pattern[i] > 0:
                i += 1
            cluster_end.append(i - 1)
        i += 1

    # Generate cluster patterns
    for start, end in zip(cluster_start, cluster_end):
        cluster_len = end - start + 1
        total_edits = int(sum(partial_pattern[start:end + 1]))
        expanded_cluster = expanded_pattern[start:end + 1]
        original_cluster_pattern = partial_pattern[start:end + 1]

        generated_patterns = generate_cluster_patterns1(total_edits, cluster_len, expanded_cluster)
        cluster_patterns.append(np.vstack([original_cluster_pattern, generated_patterns]))

    # Combine cluster patterns
    sub_deletion_patterns = combine_cluster_patterns(cluster_patterns, v, expanded_pattern)
    return np.unique(sub_deletion_patterns, axis=0)

def generate_cluster_patterns1(total_edits, cluster_length, expanded_cluster):
    # Handle the case of negative total_edits (deletions > insertions)
    if total_edits < 0:
        total_deletions = -total_edits  # The number of deletions (positive)
        valid_patterns = partition_with_limit1(total_deletions, cluster_length, deletion=True)  # Handle deletions
    else:
        valid_patterns = partition_with_limit1(total_edits, cluster_length, deletion=False)  # Handle insertions
    
    cluster_patterns = []
    for pattern in valid_patterns:
        sub_pattern = np.zeros(cluster_length, dtype=int)
        sub_pattern[np.nonzero(expanded_cluster)] = pattern[np.nonzero(expanded_cluster)]
        cluster_patterns.append(sub_pattern)
    return np.array(cluster_patterns)



def partition_with_limit1(total_edits, cluster_length, deletion=False):
    from itertools import combinations_with_replacement

    if total_edits == 0:
        return np.zeros((1, cluster_length), dtype=int)

    partitions = []
    if deletion:  # Handle deletions (negative edits)
        for combination in combinations_with_replacement(range(cluster_length), total_edits):
            pattern = np.zeros(cluster_length, dtype=int)
            for pos in combination:
                pattern[pos] = -1  # Mark deletions with -1
            partitions.append(pattern)
    else:  # Handle insertions (positive edits)
        for combination in combinations_with_replacement(range(cluster_length), total_edits):
            pattern = np.zeros(cluster_length, dtype=int)
            for pos in combination:
                pattern[pos] += 1  # Mark insertions with +1
            partitions.append(pattern)

    return np.array(partitions)

def combine_cluster_patterns(cluster_patterns, v, expanded_pattern):
    if not cluster_patterns:
        return np.array([])

    combined_patterns = cluster_patterns[0]
    for next_cluster in cluster_patterns[1:]:
        new_patterns = []
        for pattern1 in combined_patterns:
            for pattern2 in next_cluster:
                new_patterns.append(np.hstack([pattern1, pattern2]))
        combined_patterns = np.array(new_patterns)

    # Pad patterns to length v
    if combined_patterns.shape[1] < v:
        padded_patterns = np.zeros((combined_patterns.shape[0], v), dtype=int)
        for i, pattern in enumerate(combined_patterns):
            padded_patterns[i, np.nonzero(expanded_pattern)] = pattern
        combined_patterns = padded_patterns

    return combined_patterns

def generate_combinations(all_sets, l, limit=None):
    b = l + 3
    num_sets = len(all_sets)
    combinations = []

    def rec(stack, i):
        # Use outer-scope 'combinations' and 'limit'
        if limit is not None and len(combinations) >= limit:
            return                                # hard stop

        if i == num_sets:
            if not stack:
                return
            s = np.sum(np.vstack(stack), axis=0)
            if np.all(s <= b):
                combinations.append(s)
            return

        for pat in all_sets[i]:
            if limit is not None and len(combinations) >= limit:
                break                             # early exit
            rec(stack + [pat], i + 1)

    rec([], 0)
    return combinations

def decimal_to_binary_blocks(decimal_list, block_length):
    # Convert each decimal number to binary with fixed block length
    return [f"{x:0{block_length}b}" for x in decimal_list]

def check_pattern(yE, d, K, c1, c2, l, q, Par, rsDecoder, rsEncoder, N,num_blocks,markerSize):
    """
    Function to perform the quick check on a given pattern `d`.
    """
    errs = 0
    lengths = np.ones(num_blocks, dtype=int) * l 
    for i in range(1, K + c1, 2): 
        lengths[i] += markerSize

    lengths = lengths + d
    Y = divide_vector(yE, lengths, 3)
    if Y:
        Y.extend(Par)       
        non_zero_positions = [index for index, value in enumerate(d) if value != 0]
        non_zero_positions += list(range(N - c2, N))

        if np.count_nonzero(d) <= c1:
            Y = [0 if value > q - 1 else value for value in Y]
            try:
                Uhat = list(rsDecoder.decode(Y, erase_pos=non_zero_positions)[0])
            except reedsolo.ReedSolomonError:
                Uhat = None

            if Uhat:                
                Xhat = list(rsEncoder.encode(Uhat))
                
                mismatches = sum(Xhat[i] != Y[i] for i in non_zero_positions)
                if mismatches > c1 // 2:
                    errs = -1

                if Xhat[K + c1 : K + c1 + c2] == Par and errs != -1:
                    uhat_blocks = decimal_to_binary_blocks(Uhat, l)
                    uhat = [int(bit) for block in uhat_blocks for bit in block]
                    return uhat, True
    return None, False

def arrange_pattern(lst):
    result = []
    for item in lst:
        result.extend([0, item])
    return result

def find_min_levenshtein_row(A, v, d_min):
    """
    A: iterable of m rows (strings of DNA bases), each row length n_check
    v: DNA string of length n_check
    Returns (min_index_1based, min_distance)
    """
    min_dist = float('inf')
    min_idx = -1
    for i, row in enumerate(A):  
        d = levenshtein(row, v)
        if d < min_dist:
            min_dist = d
            min_idx = i
        if d <= (d_min-1) // 2:
            break
    return min_idx, min_dist

def levenshtein(a, b):
    """Compute Levenshtein distance between two sequences a and b."""
    m, n = len(a), len(b)
    D = np.zeros((m+1, n+1), dtype=int)

    for i in range(m+1):
        D[i, 0] = i
    for j in range(n+1):
        D[0, j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            D[i, j] = min(D[i-1, j] + 1,     # deletion
                          D[i, j-1] + 1,     # insertion
                          D[i-1, j-1] + cost) # substitution
    return D[m, n]
