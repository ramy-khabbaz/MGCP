import numpy as np
from reedsolo import RSCodec
import reedsolo

def GCP_Decode_binary_brute(y, n, k, l, N, K, c1, q, len_last, lim, P2, codebook_binary, d_min, opt = False):
    # Initializations
    uhat = []
    delta = len(y) - n
    c2 = 1

    # RS encoder setup
    rsEncoder = RSCodec((c1 + c2), c_exp=l)
    
    # RS decoder setup
    rsDecoder = RSCodec((c1 + c2), c_exp=l)

    # Decode check parities
    n_check = len(codebook_binary[0])          # number of DNA bases in the parity tail
    seg = y[-n_check:]                      # DNA tail to match
    idx, _ = find_min_levenshtein_row(codebook_binary, seg, d_min)
    Par = [idx]                         # decimal parity symbol(s)
    p = decimal_to_binary_blocks(Par, l)[0] # return parity bits as a string of length l
    p = list(map(int, p))   

    # Fast check
    if delta == 0:
        yE = y[:(K + c1) * l]
        lengths = np.ones(K + c1, dtype=int) * l 
        Y = divide_vector(yE, lengths)

        if Y:
            Y.extend(Par)
            ones_positions = list(range(N - c2, N))

            try:
                Uhat = list(rsDecoder.decode(Y,erase_pos=ones_positions)[0])
            except reedsolo.ReedSolomonError:
                Uhat = None

            if Uhat:
                Xhat = list(rsEncoder.encode(Uhat))
                if Xhat[K + c1 : K + c1 + c2] == Par and hamming_distance(Xhat, Y) <= c1 // 2:
                    uhat_blocks = decimal_to_binary_blocks(Uhat, l)
                    uhat = [int(bit) for block in uhat_blocks for bit in block]
                    return uhat, p
    
    #General check
    D = [0, 1]
    uhat_best = None

    for z in D:
        if delta < 0:
            adjusted_delta = delta + z
        else:
            adjusted_delta = delta - z
            
        yE = y[:k + c1 * l + adjusted_delta]
        
        if abs(adjusted_delta) < lim:
            P = int(np.sign(adjusted_delta + np.finfo(float).eps)) * P2[abs(adjusted_delta)]
    
            num_patterns = len(P)
    
            for i in range(num_patterns):
                lengths = np.ones(K + c1, dtype=int) * l
                lengths = lengths + P[i]  
    
                if np.any(lengths < 0):
                    continue
                
                Y = divide_vector(yE, lengths)
                
                if Y:
                    Y.extend(Par)
                    erasure_pattern = np.zeros(N, dtype=int)
                    erasure_pattern[N - c2:] = 1  
    
                    erasure_pattern[np.nonzero(P[i])] = 1  
    
                    ones_positions = np.where(erasure_pattern == 1)[0].tolist()
                    
                    Y = [0 if value > q - 1 else value for value in Y]
                    
                    try:
                        Uhat = list(rsDecoder.decode(Y,erase_pos=ones_positions)[0])
                        
                    except reedsolo.ReedSolomonError:
                        Uhat = None
    
                    if Uhat:
                        Xhat = list(rsEncoder.encode(Uhat))
                        
                        w = int(np.count_nonzero(erasure_pattern[:K + c1]))
                        idx_no_erasures = [j for j in range(K + c1) if erasure_pattern[j] == 0]
                        dist_no_erasures = hamming_distance(Xhat, Y, idx_no_erasures)
                        
                        if Xhat[K + c1 : K + c1 + c2] == Par and dist_no_erasures <= (c1 - w) // 2:
                            uhat_blocks = decimal_to_binary_blocks(Uhat, l)
                            uhat = [int(bit) for block in uhat_blocks for bit in block]
                            if not opt or dist_no_erasures < (c1 - w) // 2:
                                return uhat, p
                            elif uhat_best is None:
                                uhat_best = uhat
            if uhat_best is not None:
                return uhat_best, p
    return uhat, p


def GCP_Decode_binary_rep(y, n, k, l, N, K, c1, c2, q, len_last, lim, P2, t, opt = False):
    # Initializations
    uhat = []
    delta = len(y) - n

    # RS encoder setup
    rsEncoder = RSCodec((c1 + c2), c_exp=l)
    
    # RS decoder setup
    rsDecoder = RSCodec((c1 + c2), c_exp=l)

    # Decode check parities
    seg = y[-c2 * l * t:]
    p = rep_decode_gamma(seg, t, c2 * l)
    Par = binary_to_decimal_blocks(p, l) 

    # Fast check
    if delta == 0:
        yE = y[:(K + c1) * l]
        lengths = np.ones(K + c1, dtype=int) * l 
        Y = divide_vector(yE, lengths)

        if Y:
            Y.extend(Par)
            ones_positions = list(range(N - c2, N))

            try:
                Uhat = list(rsDecoder.decode(Y,erase_pos=ones_positions)[0])
            except reedsolo.ReedSolomonError:
                Uhat = None

            if Uhat:
                Xhat = list(rsEncoder.encode(Uhat))
                if Xhat[K + c1 : K + c1 + c2] == Par and hamming_distance(Xhat, Y) <= c1 // 2:
                    uhat_blocks = decimal_to_binary_blocks(Uhat, l)
                    uhat = [int(bit) for block in uhat_blocks for bit in block]
                    return uhat, p
    
    #General check
    D = [0, 1]
    uhat_best = None
    
    for z in D:
        if delta < 0:
            adjusted_delta = delta + z
        else:
            adjusted_delta = delta - z
    
        yE = y[:k + c1 * l + adjusted_delta]
        
        if abs(adjusted_delta) < lim:
            P = int(np.sign(adjusted_delta + np.finfo(float).eps)) * P2[abs(adjusted_delta)]
    
            num_patterns = len(P) 
    
            for i in range(num_patterns):
                lengths = np.ones(K + c1, dtype=int) * l
                lengths = lengths + P[i]  # Add pattern row to lengths
    
                if np.any(lengths < 0):
                    continue
                
                Y = divide_vector(yE, lengths)
                
                if Y:
                    Y.extend(Par)
                    erasure_pattern = np.zeros(N, dtype=int)  # Initialize with zeros
                    erasure_pattern[N - c2:] = 1  
    
                    # Update erasure_pattern where P[i] is nonzero
                    erasure_pattern[np.nonzero(P[i])] = 1  
    
                    ones_positions = np.where(erasure_pattern == 1)[0].tolist()
                    
                    Y = [0 if value > q - 1 else value for value in Y]
                    
                    try:
                        Uhat = list(rsDecoder.decode(Y,erase_pos=ones_positions)[0])
                        
                    except reedsolo.ReedSolomonError:
                        Uhat = None
    
                    if Uhat:
                        Xhat = list(rsEncoder.encode(Uhat))
                        
                        w = int(np.count_nonzero(erasure_pattern[:K + c1]))
                        idx_no_erasures = [j for j in range(K + c1) if erasure_pattern[j] == 0]
                        dist_no_erasures = hamming_distance(Xhat, Y, idx_no_erasures)
                        
                        if Xhat[K + c1 : K + c1 + c2] == Par and dist_no_erasures <= (c1 - w) // 2:
                            uhat_blocks = decimal_to_binary_blocks(Uhat, l)
                            uhat = [int(bit) for block in uhat_blocks for bit in block]
                            if not opt or dist_no_erasures < (c1 - w) // 2:
                                return uhat, p
                            elif uhat_best is None:
                                uhat_best = uhat
            if uhat_best is not None:
                return uhat_best, p
    return uhat, p
    #'''

def GCP_Decode_binary_localized(y, n, k, l, N, K, c1, c2, q, len_last, P1, w, W=None):
    
    c = c1 + c2
    if W is None:
        W = 3 * (w + 1)

    uhat = []
    delta = len(y) - n
    y = list(map(int, y))
    buff = [1] * (w + 1) + [0] * (w + 1) + [1] * (w + 1)

    rsEncoder = RSCodec(c, c_exp=l)
    rsDecoder = RSCodec(c, c_exp=l)

    # ---------------- Detect & correct edit errors using buffer ----------------
    if delta == 0:
        # Remove the marker of length W inserted after the first k bits
        yE = y[:]                      # copy
        del yE[k : k + W]

        lengths = np.ones(K + c, dtype=int) * l
        lengths[K - 1] = len_last

        Y = divide_vector(yE, lengths)
        

        try:
            Uhat = list(rsDecoder.decode(Y)[0])
        except reedsolo.ReedSolomonError:
            Uhat = None

        if Uhat:

            Xhat = list(rsEncoder.encode(Uhat))

            if hamming_distance(Xhat, Y) <= c // 2:
                uhat_blocks = decimal_to_binary_blocks(Uhat, l)
                uhat = [int(bit) for block in uhat_blocks for bit in block]
                if len_last < l:
                    r = k % l
                    if r != 0:
                        del uhat[(K - 1) * l : K * l - r]
                return uhat

    else:
        # With edits, try to detect the buffer tail [0^(w+1) 1^(w+1)] at its shifted location
        start = k + (w + 1) + delta          # 0-based start
        end   = k + 3 * (w + 1) + delta      # 0-based exclusive end
        seg = y[start:end] if 0 <= start <= end <= len(y) else []
        if seg == buff[(w + 1):]:
            # keep only up to k + delta bits (pre-marker content perturbed by edits)
            yE = y[: k + delta]
            # extract parity tail (last c*l bits), convert to c GF symbols
            p_tail = y[-c * l:] if len(y) >= c * l else []
            Par = binary_to_decimal_blocks(p_tail, l) if p_tail else []
        else:
            # Buffer not detected -> return first k bits as-is
            return y[:k]

    # ---------------- Primary check (delta != 0 path) ----------------
    if delta != 0:
        for i in range(len(P1)):
                lengths = np.ones(K, dtype=int) * l
                erasures = P1[i]
    
                # distribute delta across chosen erasures
                for j in range(abs(delta)):
                    m = j % c1
                    lengths[erasures[m]] += int(np.sign(delta))
    
                if np.any(lengths < 0):
                    continue
    
                Y = divide_vector(yE, lengths)
    
                if Y:
                    Y.extend(Par)
                    ones_positions = list(range(N - c2, N))
                    ones_positions.extend(erasures)
    
                    Y = [0 if val > q - 1 else val for val in Y]
    
                    try:
                        Uhat = list(rsDecoder.decode(Y, erase_pos=ones_positions)[0])
                    except reedsolo.ReedSolomonError:
                        Uhat = None
    
                    if Uhat:
                        Xhat = list(rsEncoder.encode(Uhat))
                        
                        if Xhat[K + c1 : K + c1 + c2] == Par[c1 : c1 + c2]:
                            uhat_blocks = decimal_to_binary_blocks(Uhat, l)
                            uhat = [int(bit) for block in uhat_blocks for bit in block]
                            if len_last < l:
                                r = k % l
                                if r != 0:
                                    del uhat[(K - 1) * l : K * l - r]
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

def rep_decode_gamma(y, t, k):
    arr = np.asarray(y, dtype=np.int8)
    usable = (len(arr) // t) * t
    if usable == 0:
        return [0]*k
    chunks = arr[:usable].reshape(-1, t)
    ones = chunks.sum(axis=1)
    bits = (2*ones >= t).astype(np.uint8)  # majority (ties -> 1; t is odd anyway)
    if bits.size < k:
        bits = np.pad(bits, (0, k - bits.size), mode='constant')
    else:
        bits = bits[:k]
    return bits.tolist()

def divide_vector(x, y):

    Y = []  # Final decimal result
    n = len(y)  # Number of parts

    #if sum(y) == len(x):
    parts = [None] * n  # Initialize a list to store binary parts
    start_idx = 0  # Initialize starting index

    for i in range(n):
        end_idx = start_idx + y[i]  # Calculate ending index
        part = x[start_idx:end_idx]  # Extract part from x

        if not part:  # If the part is empty
            Y.append(0)
        else:
            # Convert binary sequence to decimal
            decimal_value = int("".join(map(str, part)), 2)
            Y.append(decimal_value)

        parts[i] = part
        start_idx = end_idx  # Update starting index for the next part

    return Y

def decimal_to_binary_blocks(decimal_list, block_length):
    # Convert each decimal number to binary with fixed block length
    return [f"{x:0{block_length}b}" for x in decimal_list]

def hamming_distance(a, b, indices=None):
    """
    Compute Hamming distance between sequences a and b.
    If indices is given, compute only over those positions.
    """
    if indices is None:
        return sum(x != y for x, y in zip(a, b))
    else:
        return sum(a[i] != b[i] for i in indices)
    
def find_min_levenshtein_row(A, v, d_min):
    """
    A: iterable of m rows (strings of DNA bases), each row length n_check
    v: DNA string of length n_check
    Returns (min_index_1based, min_distance)
    """
    min_dist = float('inf')
    min_idx = -1
    for i, row in enumerate(A):  
        d = sld_suffix_distance(row, v)
        if d < min_dist:
            min_dist = d
            min_idx = i
        if d <= (d_min-1) // 2:
            break
    return min_idx, min_dist

def sld_suffix_distance(codeword: str, read: str) -> int:
    """
    Exact suffix Sequence-Levenshtein distance matching:
    min_{i=1..|A|} LD(A_rev[:i], B_rev) and min_{j=1..|B|} LD(B_rev[:j], A_rev),
    i.e., no empty truncations.

    Returns +inf if both inputs are empty (mirrors original).
    """
    A = codeword[::-1]
    B = read[::-1]
    m, n = len(A), len(B)

    # DP init: D[0][j] = j
    prev = list(range(n + 1))
    min_last_col = float('inf')  # min over D[i][n], i>=1

    for i in range(1, m + 1):
        ai = A[i - 1]
        curr = [0] * (n + 1)
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if ai == B[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost  # substitution
            )
        # Track D[i][n] for i>=1
        min_last_col = min(min_last_col, curr[n])
        prev = curr

    # After loop, prev = D[m][*]; take min over j=1..n (exclude j=0)
    if n >= 1:
        min_last_row = min(prev[1:])
    else:
        min_last_row = float('inf')

    return min(min_last_col, min_last_row)

def levenshtein(a, b):
    """Levenshtein distance between two strings a and b."""
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    # Use two rows DP
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(prev[j] + 1,           # deletion
                          curr[j - 1] + 1,       # insertion
                          prev[j - 1] + cost)    # substitution
        prev, curr = curr, prev
    return prev[n]