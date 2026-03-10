import numpy as np
import time
from multiprocessing import Pool, cpu_count
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
from pathlib import Path
from reedsolo import RSCodec
import galois
import json
import pickle
import hashlib
from mgcp.utils.loader import load_codebook_dna
from mgcp.dna.preCompute_Patterns import preCompute_Patterns
from mgcp.dna.MGCP_Decode_DNA_p2 import MGCP_Decode_DNA_p2
from mgcp.dna.GCP_Decode_DNA import GCP_Decode_DNA_brute

# -----------------------------
# Inner decoding
# -----------------------------

_global_params = {}

def init_worker(params):
    global _global_params
    _global_params = params

def decode_task(args):
    idx, consensus_seq = args
    p = _global_params  # shortcut
    
    uhat, _, _, _, _ = MGCP_Decode_DNA_p2(
        consensus_seq, p["n"], p["l_in"], p["N"], p["K"],
        p["c1"], p["c2"], p["q"], p["maxSize"], p["marker_period"],
        p["P0"], p["Pd"], p["Pi"], p["Ps"], p["codebook"], p["dmin"]
    )

    return idx, "OK", uhat, None

def decode_task_GCP(args):
    idx, consensus_seq = args
    p = _global_params  # shortcut

    uhat, _ = GCP_Decode_DNA_brute(
        consensus_seq, p["n"], p["k"], p["l_in"], p["N"], p["K"],
        p["c1"], p["q"], p["len_last"], p["lim"], p["P2"], p["codebook"], p["dmin"], p["opt"]
    )
    return idx, "OK", uhat

def parallel_decode_GCP(consensuses, n, k, l_in, N, K, c1, q, len_last, lim,
                    Patterns, codebook, dmin, opt, n_workers=None):

    if n_workers is None:
        n_workers = cpu_count()

    # pack big shared stuff once
    params = {
        "n": n, "k": k, "l_in": l_in, "N": N, "K": K,
        "c1": c1, "q": q, "len_last": len_last, "lim": lim,
        "P2": Patterns, "codebook": codebook, "dmin": dmin, "opt": opt
    }

    tasks = [(idx, seq) for idx, seq in enumerate(consensuses)]
    uhats = [None] * len(tasks)

    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker, initargs=(params,)) as executor:
        futures = [executor.submit(decode_task_GCP, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Decoding"):
            idx, status, uhat = future.result()

            uhats[idx] = uhat

    decoding_time = time.perf_counter() - start_time
    return decoding_time, uhats

def parallel_decode(consensuses,
                    n, k, l_in, N, K, c1, c2, q,
                    maxSize, marker_period, P0, Pd, Pi, Ps,
                    codebook, dmin, n_workers=None):

    if n_workers is None:
        n_workers = cpu_count()

    # shared parameters for worker processes
    params = {
        "n": n, "k": k, "l_in": l_in, "N": N, "K": K,
        "c1": c1, "c2": c2, "q": q,
        "maxSize": maxSize, "marker_period": marker_period,
        "P0": P0, "Pd": Pd, "Pi": Pi, "Ps": Ps,
        "codebook": codebook, "dmin": dmin
    }

    tasks = [(idx, seq) for idx, seq in enumerate(consensuses)]
    uhats = [None] * len(tasks)

    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=init_worker,
                             initargs=(params,)) as executor:
        futures = [executor.submit(decode_task, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Decoding"):
            idx, status, uhat, _ = future.result()
            uhats[idx] = uhat

    decoding_time = time.perf_counter() - start_time
    return decoding_time, uhats

# -----------------------------
# Outer decoding
# -----------------------------

def consensus_by_strand_id(
    uhat_list: List[np.ndarray], 
    l_out: int, 
    N_strands: int,
    seed : int
) -> List[Tuple[int, np.ndarray]]:
    """
    Majority-vote consensus per strand ID, then return (ID, payload) for valid strands.

    Parameters
    ----------
    uhat_list : list of 1-D numpy arrays of 0/1
        Each array is a binary vector. The first m bits contain metadata.
        - The first m-2 bits are the strand ID.
        - The last 2 bits are a checksum of the strand ID.
    m : int
        Number of leading bits to trim from each array. Strand ID is derived from the first m-2 bits.
    N_strands : int
        Maximum allowed strand ID. Strands with ID >= N_strands will be discarded.

    Returns
    -------
    list of (strand_id_decimal, payload_bits_ndarray)
        One entry per strand (only valid ones). Each has the decimal strand ID and the consensus payload.

    Notes
    -----
    - Ties in majority voting default to 1.
    - All arrays for a given strand ID must have the same length.
    - Strands failing the checksum test are discarded.
    """
    if l_out < 2:
        raise ValueError("m must be at least 2 to extract strand ID from first m-2 bits")

    groups = {}

    for arr in uhat_list:
        a = np.asarray(arr, dtype=np.uint8)
        if a.ndim != 1:
            raise ValueError("All inputs must be 1-D arrays.")
        if l_out > a.size:
            raise ValueError("m cannot exceed array length.")
        if not np.all((a == 0) | (a == 1)):
            raise ValueError("Arrays must be binary (0/1).")

        # Split metadata into ID
        id_bits = a[:l_out].tolist()

        rng = np.random.default_rng(seed)
        idx_map = rng.permutation(2**l_out)[:N_strands]
        fwd_map = {i: idx_map[i] for i in range(N_strands)}
        inv_map = {v: i for i, v in fwd_map.items()}

        # Compute strand ID (decimal)
        strand_id = 0
        if id_bits:
            weights = (1 << np.arange(len(id_bits) - 1, -1, -1, dtype=np.uint64))
            strand_id = inv_map.get(int(np.array(id_bits).dot(weights)),-1)

        if strand_id == -1:
            continue  # Discard this strand if ID out of range

        groups.setdefault(strand_id, []).append(a)

    # Build consensus for each strand ID
    consensus_map = {}
    for sid, arrs in groups.items():
        lengths = {arr.size for arr in arrs}
        if len(lengths) != 1:
            raise ValueError(f"All arrays for strand ID {sid} must have the same length.")
        stack = np.vstack(arrs)
        n = stack.shape[0]
        sums = stack.sum(axis=0)
        cons = (sums * 2 >= n).astype(np.uint8)  # majority vote, ties → 1
        consensus_map[sid] = cons

    # Final output: one consensus per strand ID
    out: List[Tuple[int, np.ndarray]] = []
    for sid, cons in consensus_map.items():
        payload = cons[l_out:].copy()
        out.append((sid, payload))

    return out

def consensus_by_strand_id_grs(
    uhat_list: List[np.ndarray], 
    l_out: int
) -> Tuple[List[Tuple[int, np.ndarray]], dict]:
    """
    Returns:
        output: List of (strand_id, payload) tuples
        strand_to_positions: dict mapping strand_id -> list of input positions that contributed to it
    """
    groups = {}
    first_seen = {}  # track earliest position for each strand_id
    strand_to_positions = {}  # NEW: track all positions for each strand_id

    for pos, arr in enumerate(uhat_list):
        a = np.asarray(arr, dtype=np.uint8)
        if a.ndim != 1:
            raise ValueError("All inputs must be 1-D arrays.")
        if l_out > a.size:
            raise ValueError("l_out cannot exceed array length.")
        if not np.all((a == 0) | (a == 1)):
            raise ValueError("Arrays must be binary (0/1).")

        # Split metadata into ID bits
        id_bits = a[:l_out].tolist()

        # Compute strand ID (decimal)
        strand_id = 0
        if id_bits:
            weights = (1 << np.arange(len(id_bits) - 1, -1, -1, dtype=np.uint64))
            strand_id = int(np.array(id_bits).dot(weights))

        # Track first occurrence position
        if strand_id not in first_seen:
            first_seen[strand_id] = pos
            strand_to_positions[strand_id] = []
        
        strand_to_positions[strand_id].append(pos)
        groups.setdefault(strand_id, []).append(a)

    # Build consensus for each strand ID
    consensus_map = {}
    for sid, arrs in groups.items():
        lengths = {arr.size for arr in arrs}
        if len(lengths) != 1:
            raise ValueError(f"All arrays for strand ID {sid} must have the same length.")
        stack = np.vstack(arrs)
        n = stack.shape[0]
        sums = stack.sum(axis=0)
        cons = (sums * 2 >= n).astype(np.uint8)  # majority vote, ties → 1
        consensus_map[sid] = cons

    # Final output: one consensus per strand ID
    out: List[Tuple[int, np.ndarray]] = []
    for sid, cons in consensus_map.items():
        payload = cons[l_out:].copy()
        out.append((sid, payload))

    # Sort by first_seen position to preserve cluster-size ordering
    out.sort(key=lambda x: first_seen[x[0]])

    return out, strand_to_positions

def _decode_row(args):
    """Helper function for parallel decoding of one RS codeword row."""
    row_idx, row, C, l = args
    rsDecoder = RSCodec(C, c_exp=l)
    symbols, erasures = [], []
    for idx, symbol in enumerate(row):
        if symbol == -1:
            symbols.append(0)
            erasures.append(idx)
        else:
            symbols.append(symbol)

    try:
        message = rsDecoder.decode(symbols, erase_pos=erasures)[0]
    except Exception as e:
        # Raise with row index info
        raise RuntimeError(f"row {row_idx}: {e}")
    return row_idx, message

def _decode_row_grs(args):
    """Helper function for parallel decoding of one RS codeword row."""
    row_idx, row, k, subset_idx, l = args
    try:
        message, _, _ = welch_decode_subset(row, k, l, subset_idx)
    except Exception as e:
        # Raise with row index info
        raise RuntimeError(f"row {row_idx}: {e}")
    return row_idx, message

def Outer_Decode(decoded_binary_matrix, l, C, n_workers=None):
    M, k_total = decoded_binary_matrix.shape
    num_blocks = k_total // l
    n = M
    k = n - C

    # Step 1: Convert binary rows to decimal block rows
    decimal_matrix = []
    for row in decoded_binary_matrix:
        if np.all(row == -1):  # Failed inner decoding, mark as erasure
            decimal_matrix.append([-1] * num_blocks)
        else:
            blocks = binary_to_decimal_blocks(row.tolist(), l)
            decimal_matrix.append(blocks)

    D = np.array(decimal_matrix)

    # Step 2: Transpose to get RS codewords
    D_T = D.T  # shape: (num_blocks, M)

    # Step 3: Parallel RS decoding
    tasks = [(i, D_T[i], C, l) for i in range(D_T.shape[0])]
    decoded_rows = [None] * len(tasks)

    if n_workers is None:
        n_workers = os.cpu_count()

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for row_idx, message in tqdm(
                executor.map(_decode_row, tasks, chunksize=max(1, len(tasks)//(n_workers*4))),
                total=len(tasks), desc="Decoding"
            ):
                decoded_rows[row_idx] = message
    except Exception as e:
        return None

    # Step 4: Transpose back
    decoded_matrix = np.array(decoded_rows).T  # shape: (M, k)

    # Step 5: Convert each row from decimal blocks to binary
    final_binary_matrix = []
    for row in decoded_matrix:
        binary_blocks = decimal_to_binary_blocks(row.tolist(), l)
        binary_row = [int(bit) for bin_str in binary_blocks for bit in bin_str]
        final_binary_matrix.append(binary_row)

    return np.array(final_binary_matrix)

def build_outer_decoder_input(consensus_output: List[Tuple[int, np.ndarray]], N_strands: int) -> np.ndarray:
    """
    Convert consensus output to 2D array input for Outer_Decode, filling missing rows with -1s.

    Parameters
    ----------
    consensus_output : list of (strand_id, payload)
        Output from consensus_by_strand_id.
    N_strands : int
        Total number of expected strands (rows). Missing ones will be marked with -1s.

    Returns
    -------
    np.ndarray of shape (N_strands, payload_len), dtype=int
        Matrix suitable for input to Outer_Decode.
        Failed rows are filled with -1s.
    """
    if not consensus_output:
        raise ValueError("Consensus output is empty")

    payload_len = len(consensus_output[0][1])
    result = np.full((N_strands, payload_len), -1, dtype=int)

    for strand_id, payload in consensus_output:
        if len(payload) != payload_len:
            raise ValueError("All payloads must have the same length")
        result[strand_id] = payload  # Fill the appropriate row

    return result

def verify_reconstruction(original_bytes: bytes, decoded_matrix: np.ndarray) -> None:
    """
    Verify if the decoded binary matrix reconstructs the original file.

    Parameters
    ----------
    original_bytes : bytes
        Raw file contents before encoding.
    decoded_matrix : np.ndarray
        Matrix returned by Outer_Decode, shape (M, k).

    Returns
    -------
    None
        Prints verification results.
    """
    # --- Flatten decoded matrix into a bitstream ---
    decoded_bits = decoded_matrix.flatten()

    # --- Original bits ---
    orig_bits = np.unpackbits(np.frombuffer(original_bytes, dtype=np.uint8))

    # --- Trim decoded bits to original length (removes pad) ---
    decoded_bits = decoded_bits[:orig_bits.size]

    # --- Bitwise comparison ---
    if np.array_equal(orig_bits, decoded_bits):
        print("✅ Perfect match! Original file recovered exactly.")
    else:
        num_errors = np.sum(orig_bits != decoded_bits)
        print(f"❌ Mismatch: {num_errors} bit errors out of {orig_bits.size}")
        return False

    # --- Byte-level comparison ---
    decoded_bytes = np.packbits(decoded_bits)
    if decoded_bytes.tobytes() == original_bytes:
        print("✅ Byte-for-byte identical to original.")
        return True

    else:
        print("❌ Byte mismatch (files differ).")

def gf_scalar_one(GF):
    return GF.One() if hasattr(GF, "One") else GF(1)

def gf_scalar_zero(GF):
    return GF.Zero() if hasattr(GF, "Zero") else GF(0)

def gf_zeros(GF, n):
    return GF.Zeros(n) if hasattr(GF, "Zeros") else GF(np.zeros(n, dtype=int))

def gf_ones(GF, n):
    return GF.Ones(n) if hasattr(GF, "Ones") else GF(np.ones(n, dtype=int))

def welch_decode_subset(y, k, c_exp, subset_idx=None, fcr=0, v=None, prim=None,
                        erasures_idx=None, t_hint=None):
    """
    Welch decoder that takes subset_idx instead of A.

    y: length-m received symbols (ints or GF elems)
    k: message length
    c_exp: field exponent, works over GF(2^c_exp)
    subset_idx: indices in [0, 2^c_exp - 2] selecting the m evaluation points
                if None, uses 0..m-1
    fcr: starting power, locator i is alpha^(fcr + i) before subsetting
    v: column multipliers of length m, if None uses all ones
    prim: optional irreducible polynomial
    erasures_idx: indices 0..m-1 that are known erasures
    t_hint: optional upper bound on number of errors t
    """
    # Build field
    if prim is None:
        GF = galois.GF(2**c_exp)
    else:
        GF = galois.GF(2**c_exp, irreducible_poly=prim)

    # Cast inputs
    y = GF(y)
    m = y.size
    n_parent = 2**c_exp - 1

    # Build subset indices
    if subset_idx is None:
        subset_idx = np.arange(m, dtype=int)
    else:
        subset_idx = np.asarray(subset_idx, dtype=int)
        if np.any((subset_idx < 0) | (subset_idx >= n_parent)):
            raise ValueError("subset_idx out of range")
        if subset_idx.size != m:
            raise ValueError("len(subset_idx) must equal len(y)")

    # Build locators A from subset_idx and fcr
    alpha = GF.primitive_element
    locators_all = alpha ** (fcr + np.arange(n_parent, dtype=np.int64))
    A = locators_all[subset_idx]

    # Column multipliers
    if v is None:
        v = gf_ones(GF, m)
    else:
        v = GF(v)
        if v.size != m:
            raise ValueError("len(v) must equal len(y)")

    # Handle erasures
    n = A.size
    if erasures_idx is None:
        erasures_idx = []
    erasures_idx = np.asarray(erasures_idx, dtype=int)
    if np.any((erasures_idx < 0) | (erasures_idx >= n)):
        raise ValueError("erasures_idx out of range")

    mask_keep = np.ones(n, dtype=bool)
    mask_keep[erasures_idx] = False

    # Drop erasures
    A2 = A[mask_keep]
    v2 = v[mask_keep]
    y2 = y[mask_keep]
    n2 = A2.size
    e = len(erasures_idx)

    if n2 < k:
        raise ValueError("Too many erasures. Need at least k remaining points.")

    # Scale away multipliers
    r2 = y2 / v2

    # Choose t
    if t_hint is None:
        t = max(0, (n2 - k) // 2)
    else:
        t = min(t_hint, max(0, (n2 - k) // 2))

    # Ensure enough equations
    if n2 < k + 2 * t:
        t = max(0, (n2 - k) // 2)

    deg_E = t
    deg_N = k - 1 + t

    num_unknowns = (deg_N + 1) + deg_E  # N_0..N_degN, E_1..E_degE

    # Build M (n2 x num_unknowns) and b (n2)
    M = gf_zeros(GF, (n2, num_unknowns))
    b = gf_zeros(GF, n2)

    # Power table: powers[j, i] = A2[i]^j
    max_deg = max(deg_N, deg_E)
    powers = gf_ones(GF, (max_deg + 1, n2))
    for j in range(1, max_deg + 1):
        powers[j] = powers[j - 1] * A2

    # Fill M and b
    # Columns 0..deg_N: N_j coefficients
    for j in range(deg_N + 1):
        M[:, j] = powers[j]

    # Right-hand side
    b = r2

    # Columns for E_1..E_degE: -(r_i * A_i^j)
    for j in range(1, deg_E + 1):
        col = (deg_N + 1) + (j - 1)
        M[:, col] = -(r2 * powers[j])

    # Solve M x = b over GF using forward elimination + back-substitution
    # (Gaussian elimination, no full RREF)

    # Augmented matrix [M | b]
    Aug = gf_zeros(GF, (n2, num_unknowns + 1))
    Aug[:, :num_unknowns] = M
    Aug[:, -1] = b

    rows, cols_aug = Aug.shape
    cols = cols_aug - 1  # number of unknowns
    pivot_cols = []
    pivot_count = 0
    row = 0

    # Forward elimination (only eliminate BELOW each pivot, early stop when we have enough pivots)
    for col in range(cols):
        if row >= rows:
            break

        # Find a pivot in this column at or below 'row'
        pivot_row = None
        for r in range(row, rows):
            if Aug[r, col] != 0:
                pivot_row = r
                break

        if pivot_row is None:
            continue  # no pivot in this column

        # Swap pivot row into 'row' position if needed
        if pivot_row != row:
            Aug[[row, pivot_row]] = Aug[[pivot_row, row]]

        # Normalize pivot row to make pivot 1
        inv = gf_scalar_one(GF) / Aug[row, col]
        Aug[row, :] *= inv

        # Eliminate entries below pivot (vectorized on rows below)
        if row + 1 < rows:
            below = np.where(Aug[row + 1:, col] != 0)[0]
            if below.size > 0:
                below_rows = below + (row + 1)
                factors = Aug[below_rows, col].reshape(-1, 1)
                Aug[below_rows, :] -= factors * Aug[row, :]

        pivot_cols.append(col)
        pivot_count += 1
        row += 1

        # Early stop: once we have 'num_unknowns' pivots we can stop
        if pivot_count == num_unknowns:
            break

    # Check for inconsistency: rows with all-zero coefficients but non-zero RHS
    for r in range(rows):
        if np.all(Aug[r, :cols] == 0) and Aug[r, cols] != 0:
            raise ValueError("Welch decoding failure: inconsistent linear system.")

    # Back-substitution to get a particular solution x
    x = gf_zeros(GF, num_unknowns)
    # Non-pivot (free) variables remain 0

    for i in range(pivot_count - 1, -1, -1):
        col = pivot_cols[i]
        r = i  # pivot rows are 0..pivot_count-1 after elimination
        # Solve: sum_{j >= col} Aug[r, j] * x[j] = Aug[r, cols]
        s = Aug[r, cols]
        if col + 1 < cols:
            # Vectorized sum over known x[j], j>col
            coeffs = Aug[r, col + 1:cols]
            xs = x[col + 1:cols]
            if xs.size > 0:
                s -= np.sum(coeffs * xs)
        # Pivot coefficient is 1 by construction
        x[col] = s

    # Build E and N polynomials from x
    N_coeffs = x[:deg_N + 1]
    E_coeffs = gf_zeros(GF, deg_E + 1)
    E_coeffs[0] = gf_scalar_one(GF)  # E_0 = 1 fixes the scaling
    if deg_E > 0:
        E_coeffs[1:] = x[deg_N + 1:]

    # Divide N by E to get m(x)
    try:
        N_poly = galois.Poly(N_coeffs, field=GF, order="asc")
        E_poly = galois.Poly(E_coeffs, field=GF, order="asc")
        Q, _ = divmod(N_poly, E_poly)
        m_coeffs = Q.coeffs(order="asc")
    except Exception:
        # Fallback long division (unchanged from your version)
        def trim(a):
            i = len(a) - 1
            while i > 0 and a[i] == 0:
                i -= 1
            return a[: i + 1]

        N_vec = trim(N_coeffs.copy())
        E_vec = trim(E_coeffs.copy())
        degN = len(N_vec) - 1
        degE = len(E_vec) - 1
        if degN < degE:
            m_coeffs = gf_zeros(GF, 1)
        else:
            q = gf_zeros(GF, degN - degE + 1)
            rtmp = N_vec.copy()
            while len(rtmp) - 1 >= degE and np.any(rtmp != 0):
                shift = (len(rtmp) - 1) - degE
                coef = rtmp[-1] / E_vec[-1]
                q[shift] = coef
                tmp = gf_zeros(GF, shift + len(E_vec))
                tmp[shift:] = E_vec * coef
                rtmp = trim(rtmp - tmp)
            m_coeffs = q

    # Evaluate y_hat = v * m(A)
    m_coeffs = (
        m_coeffs[:k]
        if m_coeffs.size >= k
        else np.concatenate([m_coeffs, gf_zeros(GF, k - m_coeffs.size)])
    )
    y_hat = gf_zeros(GF, A.size)
    power = gf_ones(GF, A.size)
    for coeff in m_coeffs:
        y_hat += coeff * power
        power *= A
    y_hat *= v

    err_pos = np.where(y_hat != y)[0].tolist()
    return m_coeffs[:k], y_hat, err_pos

def Outer_Decode_grs(consensus_output: List[Tuple[int, np.ndarray]], l_out, M, n_workers=None):
    if not consensus_output:
        return np.empty((0,), dtype=int), np.empty((0, 0), dtype=int)

    ids, payloads = zip(*consensus_output)            # tuple of ids, tuple of 1D arrays
    subset_idx = np.fromiter(ids, dtype=int)


    # Ensure all payloads have the same length
    first_len = len(payloads[0])
    if any(len(p) != first_len for p in payloads):
        raise ValueError("All payload vectors must have the same length")

    # Stack into a 2D array 
    binary_uhats = np.vstack(payloads).astype(int, copy=False)

    # Step 1: Convert binary rows to decimal block rows
    decimal_matrix = []
    for row in binary_uhats:
        blocks = binary_to_decimal_blocks(row.tolist(), l_out)
        decimal_matrix.append(blocks)

    D = np.array(decimal_matrix)

    # Step 2: Transpose to get RS codewords
    D_T = D.T  # shape: (num_blocks, M)    

    _ = galois.GF(2**l_out)

    # Step 3: Parallel RS decoding
    tasks = [(i, D_T[i], M, subset_idx, l_out) for i in range(D_T.shape[0])]
    decoded_rows = [None] * len(tasks)

    if n_workers is None:
        n_workers = os.cpu_count()

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for row_idx, message in tqdm(
                executor.map(_decode_row_grs, tasks, chunksize=max(1, len(tasks)//(n_workers*4))),
                total=len(tasks), desc="Decoding"
            ):
                decoded_rows[row_idx] = message
    except Exception as e:
        return None

    # Step 4: Transpose back
    decoded_matrix = np.array(decoded_rows).T  # shape: (M, k)

    # Step 5: Convert each row from decimal blocks to binary
    final_binary_matrix = []
    for row in decoded_matrix:
        binary_blocks = decimal_to_binary_blocks(row.tolist(), l_out)
        binary_row = [int(bit) for bin_str in binary_blocks for bit in bin_str]
        final_binary_matrix.append(binary_row)

    return np.array(final_binary_matrix)

# -----------------------------
# Helpers
# -----------------------------

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

def _get_cache_key(**kwargs):
    """
    Deterministic hash from the input arguments for caching precomputed patterns.
    """
    key_json = json.dumps(kwargs, sort_keys=True)
    return hashlib.sha256(key_json.encode("utf-8")).hexdigest()[:16]

def decode(file_name, input_path=None, processes = (cpu_count()//2), seed=123456):
    """
    Decode DNA file from a TXT file.
    
    Args:
        file_name (str): Name of the text file to decode (e.g., "run5.txt").
        input_path (str or Path, optional): Folder where the file is located. 
                                            Defaults to current working directory.
        processes (int): Number of processes for parallel decoding.
    """
        
    inner_decoding_time = 0.0
    total_decoding_time = 0.0
    c2 = 1
    maxSize = 1000
    Pe = 0.05
    Pd, Pi, Ps = 0.447 * Pe, 0.026 * Pe, 0.527 * Pe
    P0 = 0.25
    dmin=5
    marker_period=2
    l_in = 8
    l_out = 16
    codebook = load_codebook_dna()

    # === Load everything from JSON ===
    encoded_json_filename="encoding_params.json"
    meta_path = os.path.join(os.getcwd(), encoded_json_filename)

    # --- Load encoding parameters ---
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    params = metadata["params"]
    k = params["k"]
    n = params["n"]
    N = params["N"]
    K = params["K"]
    q = params["q"]
    l_out = params["l_out"]
    l_in = params["l_in"]
    c1 = params["inner_redundancy"]
    C = params["outer_redundancy"]
    N_strands = params["N_strands"]
    pad_size = params["pad_size"]
    filtered = params["filtered"]
    useMarker = params["useMarker"]
    original_extension = params["original_extension"]

    
    # === Handle input path ===
    if input_path is None:
        input_path = Path.cwd()
    else:
        input_path = Path(input_path)

    file_path = input_path / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"File to decode not found: {file_path}")

    # === Load consensuses and slice ===
    with file_path.open("r", encoding="utf-8") as f:
        consensuses = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(consensuses)} sequences from {file_path}")


    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Metadata file not found: {meta_path}\n"
            "Make sure you ran encode() in the same directory before decoding."
        )

    start_time = time.perf_counter()

    print("Inner decoding...")
    if c1 != 0:
        if useMarker:
            inner_decoding_time, uhats = parallel_decode(consensuses,
                n, k, l_in, N, K, c1, c2, q,
                maxSize, marker_period, P0, Pd, Pi, Ps,
                codebook, dmin, n_workers=processes
            )
        else:
            len_last = (k - 1) % l_in + 1
            lim = 6                                
            lambda_depths = [0]*lim
            lambda_depths[0] = 0
            lambda_depths[1] = 0
            lambda_depths[2] = 0

            # === caching preCompute_Patterns ===
            cache_dir = os.path.join(os.path.expanduser("~"), ".mgcp_cache")
            os.makedirs(cache_dir, exist_ok=True)

            cache_key = _get_cache_key(
                lambda_depths=lambda_depths,
                K=K,
                len_last=len_last,
                lim=lim,
                c1=c1,
            )
            cache_file = os.path.join(cache_dir, f"patterns_{cache_key}.pkl")

            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    Patterns = pickle.load(f)
                print(f"[cache] Loaded precomputed patterns from {cache_file}")
            else:
                Patterns = preCompute_Patterns(lambda_depths, K, len_last, lim, c1)
                with open(cache_file, "wb") as f:
                    pickle.dump(Patterns, f)
                print(f"[cache] Saved new precomputed patterns to {cache_file}")
                
            inner_decoding_time, uhats = parallel_decode_GCP(
                consensuses, n, k, l_in, N, K, c1, q, len_last, lim,
                Patterns, codebook, dmin, opt = False, n_workers=processes
            )
    else:
        inner_start_time = time.perf_counter()
        uhats = []

        for consensus_seq in tqdm(consensuses, total=len(consensuses), desc="Decoding"):
            if len(consensus_seq) == n:
                uhat = dna_to_binary(consensus_seq)
            else:
                uhat = None
            uhats.append(uhat)

        inner_decoding_time = time.perf_counter() - inner_start_time

    # Build mapping: track which consensus index produced each uhat
    consensus_to_uhat_map = []  # stores (consensus_idx, uhat) pairs
    for cons_idx, uhat in enumerate(uhats):
        if uhat is not None:
            consensus_to_uhat_map.append((cons_idx, uhat))
    
    uhat_list = [uhat for _, uhat in consensus_to_uhat_map]

    print("Outer decoding...")
    if filtered:
        indexed_uhat, strand_to_positions = consensus_by_strand_id_grs(uhat_list, l_out)
        #print(len(indexed_uhat))

        filter_file_path = "filter_indices.txt"  # or make this a parameter
        if os.path.exists(filter_file_path):
            with open(filter_file_path, 'r') as f:
                valid_indices = set(int(line.strip()) for line in f if line.strip())
            
            # === DIAGNOSTIC: Track wrong indices and their consensus lengths ===
            consensus_lengths = [len(seq) for seq in consensuses]
            wrong_x_diagnostics = []
            
            # For each strand_id in indexed_uhat, check if it's wrong and track all contributing consensuses
            for indexed_uhat_pos, (strand_id, payload) in enumerate(indexed_uhat):
                if strand_id not in valid_indices:
                    # This strand_id is wrong - find all consensuses that contributed to it
                    uhat_list_positions = strand_to_positions[strand_id]
                    
                    for uhat_pos in uhat_list_positions:
                        # Map back to original consensus index
                        cons_idx = consensus_to_uhat_map[uhat_pos][0]
                        cons_len = consensus_lengths[cons_idx]
                        wrong_x_diagnostics.append((cons_idx, strand_id, cons_len, n, indexed_uhat_pos))
            
            # Sort by consensus_idx before printing
            wrong_x_diagnostics.sort(key=lambda x: x[0])
            
            # Print and save diagnostics
            print(f"\n=== DIAGNOSTIC: Wrong Strand IDs ===")
            print(f"Total wrong strand occurrences: {len(wrong_x_diagnostics)}")
            print(f"Unique wrong strand IDs: {len(set(item[1] for item in wrong_x_diagnostics))}")
            print(f"Reference length (n): {n}")
            
            with open("wrong_indices_diagnostic.txt", "w") as f:
                f.write(f"consensus_idx\tstrand_id\tconsensus_length\treference_length\tlength_diff\tindexed_uhat_pos\n")
                for cons_idx, strand_id, cons_len, ref_len, indexed_uhat_pos in wrong_x_diagnostics:
                    length_diff = cons_len - ref_len
                    f.write(f"{cons_idx}\t{strand_id}\t{cons_len}\t{ref_len}\t{length_diff}\t{indexed_uhat_pos}\n")
                    
            if wrong_x_diagnostics:
                wrong_lengths = [item[2] for item in wrong_x_diagnostics]
                print(f"Wrong indices - Consensus length stats:")
                print(f"  Min: {min(wrong_lengths)}, Max: {max(wrong_lengths)}, Mean: {np.mean(wrong_lengths):.2f}")
                print(f"  Shorter than reference: {sum(1 for l in wrong_lengths if l < n)}")
                print(f"  Equal to reference: {sum(1 for l in wrong_lengths if l == n)}")
                print(f"  Longer than reference: {sum(1 for l in wrong_lengths if l > n)}")
            
            print(f"Detailed diagnostics saved to: wrong_indices_diagnostic.txt\n")
            # === END DIAGNOSTIC ===
            
            indexed_uhat_filtered = [(idx, payload) for idx, payload in indexed_uhat if idx in valid_indices]
        else:
            indexed_uhat_filtered = indexed_uhat

        #print(len(indexed_uhat_filtered))
        decoded_file = Outer_Decode_grs(indexed_uhat_filtered, l_out, N_strands-C, n_workers=processes)
    else:
        indexed_uhat = consensus_by_strand_id(uhat_list, l_out, N_strands, seed)
        #indices = [idx for idx, _ in indexed_uhat]
        decoded_binary_matrix = build_outer_decoder_input(indexed_uhat, N_strands)
        decoded_file = Outer_Decode(decoded_binary_matrix, l_out, C, processes)

    total_decoding_time = time.perf_counter() - start_time

    # === Determine success ===
    success = decoded_file is not None

    if success:
        # === Convert decoded bits back to bytes and remove padding ===
        flat_bits = np.concatenate(decoded_file).astype(np.uint8)
        if pad_size > 0:
            flat_bits = flat_bits[:-pad_size]
        decoded_bytes = np.packbits(flat_bits)

        # === Save the decoded file ===
        decoded_filename = f"decoded_output{original_extension}"
        decoded_path = os.path.join(os.getcwd(), decoded_filename)
        with open(decoded_path, "wb") as f:
            f.write(decoded_bytes.tobytes())

        print(f"Decoded file saved to {decoded_path}")
        print(f"Total decoding time:  {total_decoding_time:.2f} s")
    else:
        print("Decoding failed — No file saved")

    return success