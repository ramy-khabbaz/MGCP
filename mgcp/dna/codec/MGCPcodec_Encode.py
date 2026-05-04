import os
import numpy as np
from reedsolo import RSCodec
import math
import galois
import json
from pathlib import Path
from mgcp.dna.MGCP_Encode_DNA_p2 import MGCP_Encode_DNA_p2
from mgcp.dna.GCP_Encode_DNA import GCP_Encode_DNA_brute
from mgcp.utils.loader import load_codebook_dna

def binary_to_dna(binary_message):
    # Convert binary message to a DNA sequence
    dna_map = {'00': 'A', '01': 'T', '10': 'C', '11': 'G'}
    return "".join(dna_map["".join(map(str, binary_message[i:i+2]))] for i in range(0, len(binary_message), 2))

def binary_to_decimal_blocks(binary_message, block_length):
    # Ensure the binary message is divisible by block_length
    if len(binary_message) % block_length != 0:
        raise ValueError("The length of the binary message must be divisible by the block length.")
    
    # Group the binary message into blocks of block_length
    blocks = [binary_message[i:i + block_length] for i in range(0, len(binary_message), block_length)]
    
    # Convert each binary block to decimal
    decimal_blocks = [int("".join(map(str, block)), 2) for block in blocks]
    
    return decimal_blocks

def decimal_to_binary_blocks(decimal_list, block_length):
    # Convert each decimal number to binary with fixed block length
    return [f"{x:0{block_length}b}" for x in decimal_list]

def Outer_Encode(u, l, C):

    M, k = u.shape
    if k % l != 0:
        raise ValueError("Each row length must be divisible by the block length.")
        
    num_blocks = k // l  # number of blocks per row
    
    # Step 1: Convert each row from binary to a list of decimal blocks.
    D = np.zeros((M, num_blocks), dtype=int)
    for i in range(M):
        # Convert row (as list) to decimal blocks.
        D[i, :] = binary_to_decimal_blocks(u[i, :].tolist(), l)
    
    # Step 2: Transpose.
    D_transposed = D.T
    
    # Step 3: RS encode each row.
    rsEncoder = RSCodec(C, c_exp=l)
    encoded_rows = []
    for row in D_transposed:
        # RS encoding expects a list of integers.
        encoded_row = list(rsEncoder.encode(list(row)))
        encoded_rows.append(encoded_row)
    encoded_matrix = np.array(encoded_rows)
    
    # Step 4: Transpose back so that rows become columns.
    encoded_matrix_T = encoded_matrix.T
    
    # Step 5: Convert each decimal block back to binary.
    final_binary_matrix = []
    for row in encoded_matrix_T:
        # Convert row (list of decimals) to a list of binary strings (each of fixed length l)
        binary_strings = decimal_to_binary_blocks(row.tolist(), l)
        # Flatten each binary string into a list of integer bits
        binary_row = [int(bit) for bin_str in binary_strings for bit in bin_str]
        final_binary_matrix.append(binary_row)
    final_binary_matrix = np.array(final_binary_matrix)
    
    return final_binary_matrix

def int_to_bits(n: int, width: int):
    """MSB-first binary of n with fixed width (list of 0/1 ints)."""
    if n < 0 or n >= (1 << width):
        raise ValueError(f"int_to_bits: {n=} doesn't fit in width={width}")
    return [ (n >> (width - 1 - i)) & 1 for i in range(width) ]

def gf_zeros(GF, n):
    return GF.Zeros(n) if hasattr(GF, "Zeros") else GF(np.zeros(n, dtype=int))

def gf_ones(GF, n):
    return GF.Ones(n) if hasattr(GF, "Ones") else GF(np.ones(n, dtype=int))

def grs_encode_subset(u, c_exp=16, subset_idx=None, fcr=0, v=None, prim=None):
    """
    u: length-k ascending coefficients over GF(2^c_exp)
    subset_idx: indices from parent length n = 2^c_exp - 1; if None, use all 0..n-1
    fcr: first consecutive root b of parent cyclic RS
    v: optional column multipliers for GRS of length len(subset_idx) (or n if None)
    prim: optional irreducible polynomial
    returns y
    """
    # Field
    if prim is None:
        GF = galois.GF(2**c_exp)
    else:
        GF = galois.GF(2**c_exp, irreducible_poly=prim)

    u = GF(u)
    n = 2**c_exp - 1

    # Full set if subset_idx not provided
    if subset_idx is None:
        subset_idx = np.arange(n, dtype=int)
    else:
        subset_idx = np.asarray(subset_idx, dtype=int)
        if np.any((subset_idx < 0) | (subset_idx >= n)):
            raise ValueError("subset_idx out of range")

    # Locators and selection
    alpha = GF.primitive_element
    locators_all = alpha ** (fcr + np.arange(n, dtype=np.int64))
    A = locators_all[subset_idx]

    # Column multipliers
    if v is None:
        v = gf_ones(GF, A.size)
    else:
        v = GF(v)
        if v.size != A.size:
            raise ValueError("v length must equal len(subset_idx)")

    # Evaluate m(x) at A with Horner, then apply multipliers
    y = gf_zeros(GF, A.size)
    power = gf_ones(GF, A.size)
    for coeff in u:  # ascending powers
        y += coeff * power
        power *= A
    y *= v
    return y

def Outer_Encode_grs(u, l, subset_idx=None, fcr=0, prim=None, C=None):
    """
    Keep same interface and flow as your current Outer_Encode_grs, but make the
    encoded rows match reedsolo's format: a list of Python ints of length M + C.
    If subset_idx is None, we mirror reedsolo's default C = 2**l - 1 - M and
    take the first M + C locator positions.
    """
    M, k = u.shape
    if k % l != 0:
        raise ValueError("Each row length must be divisible by the block length.")
    num_blocks = k // l

    # Decide target length like reedsolo: n_short = M + C
    if subset_idx is None:
        if C is None:
            C = 2**l - 1 - M
        n_short = M + C
        subset_idx = np.arange(n_short, dtype=int)
    else:
        subset_idx = np.asarray(subset_idx, dtype=int)
        n_parent = 2**l - 1
        if np.any((subset_idx < 0) | (subset_idx >= n_parent)):
            raise ValueError("subset_idx out of range")
        n_short = subset_idx.size

    # Step 1: binary -> decimals
    D = np.zeros((M, num_blocks), dtype=int)
    for i in range(M):
        D[i, :] = binary_to_decimal_blocks(u[i, :].tolist(), l)

    # Step 2: transpose
    D_transposed = D.T  # shape (num_blocks, M)

    # Step 3: encode each row with grs_subset_encode
    # Important: grs_subset_encode returns (y, A, v). We only need y, and we must cast to ints.
    encoded_rows = []
    for row in D_transposed:
        y = grs_encode_subset(
            row,             # ascending coefficients if you are using coeffs
            c_exp=l,
            subset_idx=subset_idx,
            fcr=fcr,
        )
        # Convert GF elements to plain Python ints
        encoded_rows.append([int(el) for el in y])

    encoded_matrix = np.array(encoded_rows)            # shape (num_blocks, n_short)
    
    # Step 4: transpose back
    encoded_matrix_T = encoded_matrix.T                # shape (n_short, num_blocks)

    # Step 5: decimals -> binary bits
    final_binary_matrix = []
    for row in encoded_matrix_T:
        bin_strs = decimal_to_binary_blocks(row.tolist(), l)
        bits = [int(b) for s in bin_strs for b in s]
        final_binary_matrix.append(bits)

    final_binary_matrix = np.array(final_binary_matrix)  # shape (n_short, k)
    return final_binary_matrix

def calculate_best_k(max_length, c1, useMarker, l_in=8, l_out=16):
    """
    Finds the best k (multiple of l_out) for a target oligo length.
    Uses different length models depending on useMarker.
    Returns (best_k, estimated_length).
    
    Conditions:
      - l_in <= 8
      - if useMarker=True (marker_period=2), total number of l_in-blocks + c1 must be even
      - raises ValueError if no valid k satisfies all conditions and length <= max_length
    """

    # --- Basic validation ---
    if l_in > 8:
        raise ValueError(f"Invalid l_in={l_in}. It must not exceed 8.")

    def calc_length(k, c1):
        if c1 != 0:
            if useMarker:
                # Model with marker (similar to marker_period=2)
                return (16 + 4 + k + ((k * 4) / (2 * l_in)) + c1 * l_in + (c1 / 2) * 4 + 24) / 2
            else:
                # Model without marker
                return (16 + k + 24 + c1 * l_in) / 2
        else:
            return (16 + k) / 2

    best_k = None
    best_diff = float("inf")
    best_len = None

    for candidate_k in range(l_out, 20000, l_out):
        num_blocks = candidate_k // l_in

        # --- Condition for marker_period = 2 (useMarker=True) ---
        if useMarker and c1 != 0:
            total = num_blocks + c1
            if total % 2 != 0:
                # Skip invalid total parity
                continue

        current_len = calc_length(candidate_k, c1)

        # Only consider lengths <= max_length
        if current_len > max_length:
            continue

        diff = abs(current_len - max_length)
        if diff < best_diff:
            best_diff = diff
            best_k = candidate_k
            best_len = current_len

        # Optional early stop if very close
        if diff < 0.5:
            break

    # --- Raise error if no valid k found ---
    if best_k is None:
        raise ValueError(
            f"No valid encoding setting found satisfying conditions for max_length={max_length}"
        )

    return best_k, int(best_len)

def encode(file_name, max_length, inner_redundancy, outer_redundancy, input_path=None, useMarker = True, filtered = False, seed=123456):
    """
    Encode a binary file into DNA sequences using DNA-MGC+ codec. The max_length parameter sets the target oligo length, which will be matched as closely as possible. The inner_redundancy parameters controls the number of guess parities generated by the inner MGC+ code. The outer_redundancy parameter controls the number of additional sequences generated by the outer RS code. The use_marker option enables the inclusion of periodic markers. If the filtered option is enabled, the encoder will generate 2^16 candidate sequences from which any subset of the desired size can be chosen to represent the file through separate filtering. The indices are randomized using the provided seed for reproducibility. The output is a text file containing the encoded DNA sequences.

    Args:
        file_name (str): Name of the input file to encode (e.g., "data.bin").
        input_path (str or Path, optional): Folder where the file is located. Defaults to cwd.
        max_length (int): Max length of each oligo.
        inner_redundancy (int): Inner redundancy parameter.
        outer_redundancy (int): Outer redundancy parameter.
        useMarker (bool): Use marker-based encoding.
        filtered (bool): Whether to filter the outer encoding subset.
        seed (int): Random seed for reproducibility.
    """

    # === Fixed parameters ===
    l_in = 8
    l_out = 16
    MAX_STRANDS = 1 << l_out

    # === Handle input path ===
    if input_path is None:
        input_path = Path.cwd()
    else:
        input_path = Path(input_path)

    input_file_path = input_path / file_name
    if not input_file_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
    file_extension = input_file_path.suffix

    print(f"Encoding file: {input_file_path}")

    # === Load codebook ===
    codebook = load_codebook_dna()

    # === Compute best k ===
    k, est_len = calculate_best_k(max_length, inner_redundancy, useMarker, l_in, l_out)
    print(f"Information bits length k = {k}b, oligo length = {est_len} NTs")

    # === Load bits from input file and build u (M inferred from file size) ===
    data_bytes = input_file_path.read_bytes()  # read as bytes, do NOT unzip
    byte_arr = np.frombuffer(data_bytes, dtype=np.uint8)
    bits = np.unpackbits(byte_arr)

    # Compute M automatically from total bits and fixed k
    M = math.ceil(bits.size / k)

    # CASE 1: File too large (information strands exceed 2^l_out)
    if M > MAX_STRANDS:
        raise ValueError(
            f"\nFile too large for current configuration.\n"
            f"This encoder supports at most {MAX_STRANDS} information strands (2^{l_out}).\n"
            f"Your file requires M = {M} strands.\n\n"
            f"Please split the file into smaller packets and encode each packet separately.\n"
            f"Each packet must produce at most {MAX_STRANDS} strands.\n"
            f"You will then need to manage packet indexing and metadata externally "
            f"(e.g., add packet IDs and reassemble after decoding)."
        )
    
    # CASE 2: Too much outer redundancy
    if M + outer_redundancy > MAX_STRANDS:
        raise ValueError(
            f"Too much outer redundancy: encoded strands + outer_redundancy = {M + outer_redundancy} "
            f"exceeds maximum {MAX_STRANDS}. Reduce outer_redundancy to ≤ {MAX_STRANDS - M}."
        )
    
    print(f"Number of oligos to encode: M = {M}")

    # Pad ONLY the last row with RANDOM bits
    total_needed = M * k
    rng = np.random.default_rng()
    pad_size = total_needed - bits.size
    if pad_size > 0:
        pad = rng.integers(0, 2, size=pad_size, dtype=np.uint8)
        bits = np.concatenate([bits, pad])

    # Reshape into (M, k) binary matrix
    u = bits.reshape(M, k)

    # Outer encode
    if filtered:
        subset_idx = np.sort(rng.choice(2**l_out-1, size=M+outer_redundancy, replace=False))
        u_outer = Outer_Encode_grs(u, l_out, subset_idx)
    else:
        u_outer = Outer_Encode(u, l_out, outer_redundancy)
    N_strands = len(u_outer)

    # sanity: ensure we have enough index space
    if N_strands > (1 << l_out):
        raise ValueError(f"Need l_out >= ceil(log2(N_strands)); got l_out={l_out}, N_strands={N_strands}")

    encoded_strands = []
    barcodes = []

    rng = np.random.default_rng(seed)
    idx_map = rng.permutation(2**l_out)[:(M+outer_redundancy)]
    fwd_map = {i: idx_map[i] for i in range(M+outer_redundancy)}

    # Inner encode
    if inner_redundancy == 0:
        N = K = q = None
        for idx, row in enumerate(u_outer):
            x = binary_to_dna(row)
            # l_out-bit index (MSB-first)
            if filtered:
                idx_bits = int_to_bits(subset_idx[idx], l_out)
            else:
                idx_bits = int_to_bits(fwd_map[idx], l_out)

            # row is a numpy array of 0/1; turn prefix into np.array, then concatenate
            prefix = np.array(idx_bits, dtype=row.dtype)
            bc = binary_to_dna(prefix)
            barcodes.append(bc)
            
            # prepend onto the encoded protein string
            X_bc = bc + x
            encoded_strands.append(X_bc)
        encoded_file = np.array(encoded_strands)
        n = len(encoded_file[0])
    else:
        for idx, row in enumerate(u_outer):
            # l_out-bit index (MSB-first)
            if filtered:
                idx_bits = int_to_bits(subset_idx[idx], l_out)
            else:
                idx_bits = int_to_bits(fwd_map[idx], l_out)

            # row is a numpy array of 0/1; turn prefix into np.array, then concatenate
            prefix = np.array(idx_bits, dtype=row.dtype)
            barcodes.append(binary_to_dna(prefix))
            row_with_prefix = np.concatenate([prefix, row])

            # MGCP/GCP encode the augmented row
            if useMarker:
                x, n, N, K, q, U, X = MGCP_Encode_DNA_p2(row_with_prefix, l_in, inner_redundancy, codebook)
            else:
                x, n, N, K, q, U, X = GCP_Encode_DNA_brute(row_with_prefix, l_in, inner_redundancy, codebook)

            encoded_strands.append(x)  # x is your encoded protein/DNA string
        encoded_file = np.array(encoded_strands)

        
    # === Save encoded results ===
    output_dict = {
        "params": {
            "M": M,
            "k": k,
            "N_strands": N_strands,
            "n": int(n) if n is not None else None,
            "N": int(N) if N is not None else None,
            "K": int(K) if K is not None else None,
            "q": q,
            "l_out": l_out,
            "l_in": l_in,
            "inner_redundancy": inner_redundancy,
            "outer_redundancy": outer_redundancy,
            "pad_size": pad_size,
            "original_extension": file_extension,
            "useMarker": useMarker,
            "filtered": filtered,
        }
    }

    # Write .txt file with all strands
    txt_filename = "encoded_file.txt"
    json_filename = "encoding_params.json"
    txt_path = os.path.join(os.getcwd(), txt_filename)
    json_path = os.path.join(os.getcwd(), json_filename)

    with open(txt_path, "w", encoding="utf-8") as f:
        for strand in encoded_strands:
            f.write(f"{strand}\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)

    print(f"Encoded file saved to {txt_path}")
    print(f"Encoding parameters saved to {json_path}")

    return encoded_file
