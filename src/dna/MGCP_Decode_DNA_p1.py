import numpy as np
from reedsolo import RSCodec
import reedsolo
import math

def MGCP_Decode_DNA_p1(y, n, k, l, N, K, c1, c2, q, t, L, maxSize):
    # Initializations
    uhat = []
    delta = len(y) - n
    size_sub = 0
    final_sub_deletion_patterns1 = []
    d = []

    # RS encoder setup
    rsEncoder = RSCodec((c1 + c2), c_exp=l)
    
    # RS decoder setup
    rsDecoder = RSCodec((c1 + c2), c_exp=l)

    # Convert DNA to binary
    y_bin = dna_to_binary(y)

    # Decode check parities
    seg = y_bin[-c2 * l * t:]
    p = rep_decode_gamma(seg, t, c2 * l)
    Par = binary_to_decimal_blocks(p, l)

    # Fast check
    if delta == 0:
        size_sub += 1
        yE = y_bin[:(K + c1) * (l + 4)]
        lengths = np.ones(K + c1, dtype=int) * (l + 4)
        Y = divide_vector(yE, lengths, 4)

        if Y:
            Y.extend(Par)
            ones_positions = list(range(N - c2, N))

            try:
                Uhat = list(rsDecoder.decode(Y,erase_pos=ones_positions)[0])
            except reedsolo.ReedSolomonError as e:
                Uhat = None

            if Uhat:
                Xhat = list(rsEncoder.encode(Uhat))
                if Xhat[K + c1 : K + c1 + c2] == Par:
                    uhat_blocks = decimal_to_binary_blocks(Uhat, l)
                    uhat = [int(bit) for block in uhat_blocks for bit in block]
                    return uhat 

    # Trellis Calculation

    #D = np.arange(-3, 4)
    #D = np.sort(np.abs(D))    
    #for z in D:
    #delta = len(y) - n + z

    delta = len(y) - n
    yE = y[:(K + c1) * (l // 2 + 2) + delta]
    d = deletion_error_location(yE, L, K, delta, l, 2, c1)
    d1 = d
    
    # Phase 1: Quick Check
    lengths = np.ones(K + c1, dtype=int) * (l // 2 + 2)
    lengths = lengths + d
    Y = divide_vector_dna(yE, lengths, 2)
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
                if Xhat[K + c1 : K + c1 + c2] == Par:
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
    final_sub_deletion_patterns1 = generate_combinations(all_sub_deletion_patterns1, l)
    size_sub += len(final_sub_deletion_patterns1)

    # Check all patterns generated after local apportionment
    for pattern_idx in range(min(len(final_sub_deletion_patterns1), maxSize)):
        d2 = final_sub_deletion_patterns1[pattern_idx].astype(int).tolist()
        if not d2:
            continue
        uhat, valid = check_pattern(yE, d2, K, c1, c2, l, q, Par, rsDecoder, rsEncoder, N)
        if valid:
            return uhat
          
    # Phase 3: Last Try
    while sum(d1) != delta:
        # Start from the end and work backwards
        for i in range(len(d1) - 1, -1, -1):
            if d1[i] != 0:
                d1[i] = 0  # Set element to zero
                
            # Check if sum is now equal to delta
            if sum(d1) == delta:
                lengths = np.ones(K + c1, dtype=int) * (l // 2 + 2)
                lengths = lengths + d1
                Y = divide_vector_dna(yE, lengths, 2)
                size_sub += 1
                if Y:
                    Y.extend(Par)
                    erasure_pattern = [0] * (N - c2) + [1] * (c2)
                    for i in range(len(d1)):
                        if d1[i] != 0:
                            erasure_pattern[i] = 1

                    ones_positions = [i for i, value in enumerate(erasure_pattern) if value == 1]

                    if np.count_nonzero(d1) <= c1:
                        Y = [0 if value > q - 1 else value for value in Y]
                        try:
                            Uhat = list(rsDecoder.decode(Y,erase_pos=ones_positions)[0])
                        except reedsolo.ReedSolomonError as e:
                            Uhat = None

                        if Uhat:                
                            Xhat = list(rsEncoder.encode(Uhat))
                            if Xhat[K + c1 : K + c1 + c2] == Par:
                                uhat_blocks = decimal_to_binary_blocks(Uhat, l)
                                uhat = [int(bit) for block in uhat_blocks for bit in block]
                                return uhat
        
                # Local Apportionment
                consecutive_sequences = []
                current_sequence = []

                # Find consecutive sequences in d1
                for i in range(len(d1)):
                    if d1[i] != 0:
                        current_sequence.append(i)
                    elif current_sequence:
                        consecutive_sequences.append(current_sequence)
                        current_sequence = []
                if current_sequence:
                    consecutive_sequences.append(current_sequence)

                # Create partial sub-deletion patterns for each consecutive sequence
                partial_sub_deletion_patterns = []
                for current_sequence in consecutive_sequences:
                    partial_sub_deletion_pattern = np.zeros(len(d1))
                    partial_sub_deletion_pattern[current_sequence] = np.take(d1, current_sequence)
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
                final_sub_deletion_patterns1 = generate_combinations(all_sub_deletion_patterns1, l)
                size_sub += len(final_sub_deletion_patterns1)

                # Check all patterns generated after local apportionment
                for pattern_idx in range(min(len(final_sub_deletion_patterns1), maxSize)):
                    d1 = final_sub_deletion_patterns1[pattern_idx].astype(int).tolist()
                    if not d1:
                        continue
                    uhat, valid = check_pattern(yE, d1, K, c1, c2, l, q, Par, rsDecoder, rsEncoder, N)
                    if valid:
                        return uhat
                
            # If the pattern becomes all zeros and we still don't have sum(d) == delta
            if all(value == 0 for value in d1) and sum(d1) != delta:
                #break
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
        end_idx = start_idx + y[i]  # Calculate ending index
        part = x[start_idx:end_idx - marker_size]  # Extract part from x

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
def deletion_error_location(r, L, v, delta, block_size, l, c1):

    v += c1

    max_shift_per_block = 2
    max_delta = abs(delta) + 2

    P = np.zeros((v + 1, 2 * max_delta + 1))
    Q = np.zeros((v + 1, 2 * max_delta + 1))
    P[0, max_delta] = 1  # Initial state with zero shift

    # Phase 1: Compute likelihoods and record preceding states
    for i in range(1, v + 1):
        for omega in range(-max_delta, max_delta + 1):

            if i==v and omega != delta:
                continue

            current_m = decode_rib(r, i - 1, omega, block_size // 2, l)

            if current_m == -1:
                continue

            m_idx = map_dna_to_number(current_m)

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
    Z[-1] = delta

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

# Map DNA sequence to a number
def map_dna_to_number(dna_seq):

    mapping = {
        "AC": 0,
        "AT": 1, "AG": 1,
        "AA": 2,
        "CC": 3, "TC": 3, "GC": 3,
        "CT": 4, "CG": 4, "TT": 4, "TG": 4, "GT": 4, "GG": 4,
        "CA": 5, "TA": 5, "GA": 5
    }
    return mapping.get(dna_seq, -1)

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

def generate_combinations(all_sub_deletion_patterns, l):
    b = l + 4
    num_sets = len(all_sub_deletion_patterns)
    combinations = []

    def recursive_combinations(current_combination, set_idx):
        if set_idx >= num_sets:
            if not current_combination:  # Avoid stacking an empty combination
                return
            sum_pattern = np.sum(np.vstack(current_combination), axis=0)
            if np.all(sum_pattern <= b):
                combinations.append(sum_pattern)
            return

        for pattern in all_sub_deletion_patterns[set_idx]:
            new_combination = list(current_combination)
            new_combination.append(pattern)
            recursive_combinations(new_combination, set_idx + 1)

    recursive_combinations([], 0)
    return combinations

# DNA to binary conversion
def dna_to_binary(dna_seq):
    binary_seq = []
    for base in dna_seq:
        if base == 'A':
            binary_seq.extend([0, 0])
        elif base == 'T':
            binary_seq.extend([0, 1])
        elif base == 'C':
            binary_seq.extend([1, 0])
        elif base == 'G':
            binary_seq.extend([1, 1])
        else:
            raise ValueError(f"Invalid DNA character: {base}")
    return binary_seq

# Divide vector for DNA sequences
def divide_vector_dna(x, y, marker_size):

    Y = []  # Final decimal result
    n = len(y)  # Number of parts

    parts = [None] * n  # Initialize a list to store DNA parts
    start_idx = 0  # Initialize starting index

    for i in range(n):
        end_idx = start_idx + y[i]  # Calculate ending index
        part = x[start_idx:end_idx - marker_size]  # Extract part from x

        if not part:  # If the part is empty
            Y.append(0)
        else:
            # Convert DNA sequence to binary
            binary_seq = dna_to_binary(part)

            # Convert binary sequence to decimal
            decimal_value = int("".join(map(str, binary_seq)), 2)
            Y.append(decimal_value)

        parts[i] = part
        start_idx = end_idx  # Update starting index for the next part

    return Y


def decimal_to_binary_blocks(decimal_list, block_length):
    # Convert each decimal number to binary with fixed block length
    return [f"{x:0{block_length}b}" for x in decimal_list]

def check_pattern(yE, d, K, c1, c2, l, q, Par, rsDecoder, rsEncoder, N):

    lengths = np.ones(K + c1, dtype=int) * (l // 2 + 2)
    lengths += d
    Y = divide_vector_dna(yE, lengths, 2)
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
                if Xhat[K + c1 : K + c1 + c2] == Par:
                    uhat_blocks = decimal_to_binary_blocks(Uhat, l)
                    uhat = [int(bit) for block in uhat_blocks for bit in block]
                    return uhat, True
    return None, False