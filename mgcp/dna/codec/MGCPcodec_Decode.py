import numpy as np
import time
from typing import Tuple, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import math
from pathlib import Path
from reedsolo import RSCodec
import galois
import json
import pickle
import hashlib
from mgcp.utils.loader import load_codebook_dna
from mgcp.dna.preCompute_Patterns import preCompute_Patterns
from mgcp.dna.GCP_Decode_DNA import GCP_Decode_DNA_brute
from mgcp.dna.decode import decode as decode_dna
from mgcp.dna.codec.multipacket_utils import (
    FILTERED_CANDIDATE_POOL_SIZE,
    MAX_SEQS_PER_PACKET,
    maximize_outer_redundancy_for_filtered_packets,
    plan_packetization_by_rate,
    rate_to_redundancy,
)

# -----------------------------
# Inner decoding
# -----------------------------

_global_params = {}

def init_worker(params):
    global _global_params
    _global_params = params

def decode_task_GCP(args):
    idx, consensus_seq = args
    p = _global_params
    uhat, _ = GCP_Decode_DNA_brute(
        consensus_seq, p["n"], p["k"], p["l_in"], p["N"], p["K"],
        p["c1"], p["q"], p["len_last"], p["lim"], p["P2"], p["codebook"], p["dmin"], p["opt"]
    )
    return idx, "OK", uhat


def _decode_realtime_task(args):
    idx, consensus_seq, metadata = args
    try:
        uhat, diagnostics = decode_dna(
            consensus_seq,
            metadata=metadata,
            return_diagnostics=True,
        )
    except Exception as exc:
        return idx, None, {
            "error": f"{type(exc).__name__}: {exc}",
            "selected_profile": None,
            "estimated_Pe": None,
        }
    return idx, uhat, diagnostics


def _summarize_realtime_diagnostics(diagnostics):
    summary = {
        "attempted": len(diagnostics),
        "decoded": 0,
        "failed": 0,
        "avg_Pd": None,
        "avg_Pi": None,
        "avg_Ps": None,
        "avg_Pe": None,
        "avg_drift_rate": None,
    }
    if not diagnostics:
        return summary

    profiles = []
    pe_values = []
    drift_rates = []
    for item in diagnostics:
        profile = item.get("selected_profile")
        if profile is None:
            summary["failed"] += 1
            continue
        summary["decoded"] += 1
        profiles.append((profile["Pd"], profile["Pi"], profile["Ps"]))
        pe_values.append(item.get("estimated_Pe"))
        if item.get("drift_rate") is not None:
            drift_rates.append(item["drift_rate"])

    if profiles:
        summary["avg_Pd"] = sum(profile[0] for profile in profiles) / len(profiles)
        summary["avg_Pi"] = sum(profile[1] for profile in profiles) / len(profiles)
        summary["avg_Ps"] = sum(profile[2] for profile in profiles) / len(profiles)
    clean_pe_values = [value for value in pe_values if value is not None]
    if clean_pe_values:
        summary["avg_Pe"] = sum(clean_pe_values) / len(clean_pe_values)
    if drift_rates:
        summary["avg_drift_rate"] = sum(drift_rates) / len(drift_rates)
    return summary


def _normalize_worker_count(requested=None):
    """Return a portable, positive worker count capped by available CPUs."""
    available = os.cpu_count() or 1
    if requested is None:
        requested = max(1, available // 2)
    try:
        requested = int(requested)
    except (TypeError, ValueError):
        requested = max(1, available // 2)
    return max(1, min(requested, available))


def _cap_workers_for_tasks(n_workers, task_count):
    if task_count <= 0:
        return 1
    return max(1, min(_normalize_worker_count(n_workers), task_count))


def _plan_multipacket_workers(total_workers, num_packets):
    """
    Split a machine-wide worker budget between packet-level and per-packet work.
    Whole-packet parallelism is preferred; spare budget goes to inner/outer decode.
    """
    total_workers = _normalize_worker_count(total_workers)
    num_packets = max(1, int(num_packets or 1))
    packet_workers = min(total_workers, num_packets)
    per_packet_workers = max(1, total_workers // packet_workers)
    return packet_workers, per_packet_workers


def parallel_decode_GCP(consensuses, n, k, l_in, N, K, c1, q, len_last, lim,
                        Patterns, codebook, dmin, opt, n_workers=None, show_progress=True):
    params = {
        "n": n, "k": k, "l_in": l_in, "N": N, "K": K,
        "c1": c1, "q": q, "len_last": len_last, "lim": lim,
        "P2": Patterns, "codebook": codebook, "dmin": dmin, "opt": opt
    }
    tasks = [(idx, seq) for idx, seq in enumerate(consensuses)]
    uhats = [None] * len(tasks)
    n_workers = _cap_workers_for_tasks(n_workers, len(tasks))

    start_time = time.perf_counter()
    if n_workers == 1:
        for idx, consensus_seq in tqdm(tasks, total=len(tasks), desc="Decoding", disable=not show_progress):
            uhat, _ = GCP_Decode_DNA_brute(
                consensus_seq, n, k, l_in, N, K,
                c1, q, len_last, lim, Patterns, codebook, dmin, opt
            )
            uhats[idx] = uhat
    else:
        chunksize = max(1, len(tasks) // (n_workers * 8))
        with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker, initargs=(params,)) as executor:
            for idx, status, uhat in tqdm(
                executor.map(decode_task_GCP, tasks, chunksize=chunksize),
                total=len(tasks), desc="Decoding", disable=not show_progress):
                uhats[idx] = uhat

    decoding_time = time.perf_counter() - start_time
    return decoding_time, uhats

def parallel_decode_realtime(consensuses,
                             n, l_in, N, K, c1, marker_period, q,
                             n_workers=None, show_progress=True):
    metadata = {
        "l": l_in,
        "parities_count": c1,
        "marker_period": marker_period,
        "k": K * l_in,
        "K": K,
        "n": n,
        "N": N,
        "q": q,
    }
    tasks = [
        (idx, seq, metadata)
        for idx, seq in enumerate(consensuses)
    ]
    uhats = [None] * len(tasks)
    diagnostics = [None] * len(tasks)
    n_workers = _cap_workers_for_tasks(n_workers, len(tasks))

    start_time = time.perf_counter()
    if n_workers == 1:
        results = map(_decode_realtime_task, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=n_workers)
        chunksize = max(1, len(tasks) // (n_workers * 8))
        results = executor.map(_decode_realtime_task, tasks, chunksize=chunksize)

    try:
        for idx, uhat, item_diagnostics in tqdm(
            results,
            total=len(tasks),
            desc="Decoding",
            disable=not show_progress,
        ):
            uhats[idx] = uhat
            diagnostics[idx] = item_diagnostics
    finally:
        if n_workers != 1:
            executor.shutdown()

    decoding_time = time.perf_counter() - start_time
    return decoding_time, uhats, _summarize_realtime_diagnostics(diagnostics)

# -----------------------------
# Outer decoding
# -----------------------------

def consensus_by_strand_id(
    uhat_list: List[np.ndarray],
    l_out: int,
    N_strands: int,
    seed: int
) -> List[Tuple[int, np.ndarray]]:
    if l_out < 2:
        raise ValueError("m must be at least 2 to extract sequence ID from first m-2 bits")

    rng = np.random.default_rng(seed)
    idx_map = rng.permutation(2**l_out)[:N_strands]
    fwd_map = {i: idx_map[i] for i in range(N_strands)}
    inv_map = {v: i for i, v in fwd_map.items()}

    groups = {}

    for arr in uhat_list:
        a = np.asarray(arr, dtype=np.uint8)
        if a.ndim != 1:
            raise ValueError("All inputs must be 1-D arrays.")
        if l_out > a.size:
            raise ValueError("m cannot exceed array length.")
        if not np.all((a == 0) | (a == 1)):
            raise ValueError("Arrays must be binary (0/1).")

        id_bits = a[:l_out].tolist()
        strand_id = 0
        if id_bits:
            weights = (1 << np.arange(len(id_bits) - 1, -1, -1, dtype=np.uint64))
            strand_id = inv_map.get(int(np.array(id_bits).dot(weights)), -1)

        if strand_id == -1:
            continue

        groups.setdefault(strand_id, []).append(a)

    consensus_map = {}
    for sid, arrs in groups.items():
        lengths = {arr.size for arr in arrs}
        if len(lengths) != 1:
            raise ValueError(f"All arrays for sequence ID {sid} must have the same length.")
        stack = np.vstack(arrs)
        n = stack.shape[0]
        sums = stack.sum(axis=0)
        cons = (sums * 2 >= n).astype(np.uint8)
        consensus_map[sid] = cons

    out: List[Tuple[int, np.ndarray]] = []
    for sid, cons in consensus_map.items():
        payload = cons[l_out:].copy()
        out.append((sid, payload))

    return out

def consensus_by_strand_id_grs(
    uhat_list: List[np.ndarray],
    l_out: int
) -> Tuple[List[Tuple[int, np.ndarray]], dict]:
    groups = {}
    first_seen = {}
    strand_to_positions = {}

    for pos, arr in enumerate(uhat_list):
        a = np.asarray(arr, dtype=np.uint8)
        if a.ndim != 1:
            raise ValueError("All inputs must be 1-D arrays.")
        if l_out > a.size:
            raise ValueError("l_out cannot exceed array length.")
        if not np.all((a == 0) | (a == 1)):
            raise ValueError("Arrays must be binary (0/1).")

        id_bits = a[:l_out].tolist()
        strand_id = 0
        if id_bits:
            weights = (1 << np.arange(len(id_bits) - 1, -1, -1, dtype=np.uint64))
            strand_id = int(np.array(id_bits).dot(weights))

        if strand_id not in first_seen:
            first_seen[strand_id] = pos
            strand_to_positions[strand_id] = []

        strand_to_positions[strand_id].append(pos)
        groups.setdefault(strand_id, []).append(a)

    consensus_map = {}
    for sid, arrs in groups.items():
        lengths = {arr.size for arr in arrs}
        if len(lengths) != 1:
            raise ValueError(f"All arrays for sequence ID {sid} must have the same length.")
        stack = np.vstack(arrs)
        n = stack.shape[0]
        sums = stack.sum(axis=0)
        cons = (sums * 2 >= n).astype(np.uint8)
        consensus_map[sid] = cons

    out: List[Tuple[int, np.ndarray]] = []
    for sid, cons in consensus_map.items():
        payload = cons[l_out:].copy()
        out.append((sid, payload))

    out.sort(key=lambda x: first_seen[x[0]])
    return out, strand_to_positions

def _decode_row(args):
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
        raise RuntimeError(f"row {row_idx}: {e}")
    return row_idx, message

def _decode_row_grs(args):
    row_idx, row, k, subset_idx, l = args
    try:
        message, _, _ = welch_decode_subset(row, k, l, subset_idx)
    except Exception as e:
        raise RuntimeError(f"row {row_idx}: {e}")
    return row_idx, message

def Outer_Decode(decoded_binary_matrix, l, C, n_workers=None, show_progress=True):
    M, k_total = decoded_binary_matrix.shape
    num_blocks = k_total // l

    decimal_matrix = []
    for row in decoded_binary_matrix:
        if np.all(row == -1):
            decimal_matrix.append([-1] * num_blocks)
        else:
            blocks = binary_to_decimal_blocks(row.tolist(), l)
            decimal_matrix.append(blocks)

    D = np.array(decimal_matrix)
    D_T = D.T

    tasks = [(i, D_T[i], C, l) for i in range(D_T.shape[0])]
    decoded_rows = [None] * len(tasks)
    n_workers = _cap_workers_for_tasks(n_workers, len(tasks))

    try:
        if n_workers == 1:
            for task in tqdm(tasks, total=len(tasks), desc="Decoding", disable=not show_progress):
                row_idx, message = _decode_row(task)
                decoded_rows[row_idx] = message
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for row_idx, message in tqdm(
                    executor.map(_decode_row, tasks, chunksize=max(1, len(tasks) // (n_workers * 4))),
                    total=len(tasks), desc="Decoding", disable=not show_progress
                ):
                    decoded_rows[row_idx] = message
    except Exception:
        try:
            for task in tqdm(tasks, total=len(tasks), desc="Decoding", disable=not show_progress):
                row_idx, message = _decode_row(task)
                decoded_rows[row_idx] = message
        except Exception:
            return None

    decoded_matrix = np.array(decoded_rows).T

    final_binary_matrix = []
    for row in decoded_matrix:
        binary_blocks = decimal_to_binary_blocks(row.tolist(), l)
        binary_row = [int(bit) for bin_str in binary_blocks for bit in bin_str]
        final_binary_matrix.append(binary_row)

    return np.array(final_binary_matrix)

def build_outer_decoder_input(consensus_output: List[Tuple[int, np.ndarray]], N_strands: int) -> np.ndarray:
    if not consensus_output:
        raise ValueError("Consensus output is empty")

    payload_len = len(consensus_output[0][1])
    result = np.full((N_strands, payload_len), -1, dtype=int)

    for strand_id, payload in consensus_output:
        if len(payload) != payload_len:
            raise ValueError("All payloads must have the same length")
        result[strand_id] = payload

    return result

def verify_reconstruction(original_bytes: bytes, decoded_matrix: np.ndarray) -> None:
    decoded_bits = decoded_matrix.flatten()
    orig_bits = np.unpackbits(np.frombuffer(original_bytes, dtype=np.uint8))
    decoded_bits = decoded_bits[:orig_bits.size]

    if np.array_equal(orig_bits, decoded_bits):
        print("✅ Perfect match! Original file recovered exactly.")
    else:
        num_errors = np.sum(orig_bits != decoded_bits)
        print(f"❌ Mismatch: {num_errors} bit errors out of {orig_bits.size}")
        return False

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
    if prim is None:
        GF = galois.GF(2**c_exp)
    else:
        GF = galois.GF(2**c_exp, irreducible_poly=prim)

    y = GF(y)
    m = y.size
    n_parent = 2**c_exp

    if subset_idx is None:
        subset_idx = np.arange(m, dtype=int)
    else:
        subset_idx = np.asarray(subset_idx, dtype=int)
        if np.any((subset_idx < 0) | (subset_idx >= n_parent)):
            raise ValueError("subset_idx out of range")
        if subset_idx.size != m:
            raise ValueError("len(subset_idx) must equal len(y)")

    alpha = GF.primitive_element
    locators_all = np.concatenate([
        alpha ** (fcr + np.arange(n_parent - 1, dtype=np.int64)),
        gf_zeros(GF, 1),
    ])
    A = locators_all[subset_idx]

    if v is None:
        v = gf_ones(GF, m)
    else:
        v = GF(v)
        if v.size != m:
            raise ValueError("len(v) must equal len(y)")

    n = A.size
    if erasures_idx is None:
        erasures_idx = []
    erasures_idx = np.asarray(erasures_idx, dtype=int)
    if np.any((erasures_idx < 0) | (erasures_idx >= n)):
        raise ValueError("erasures_idx out of range")

    mask_keep = np.ones(n, dtype=bool)
    mask_keep[erasures_idx] = False

    A2 = A[mask_keep]
    v2 = v[mask_keep]
    y2 = y[mask_keep]
    n2 = A2.size
    e = len(erasures_idx)

    if n2 < k:
        raise ValueError("Too many erasures. Need at least k remaining points.")

    r2 = y2 / v2

    if t_hint is None:
        t = max(0, (n2 - k) // 2)
    else:
        t = min(t_hint, max(0, (n2 - k) // 2))

    if n2 < k + 2 * t:
        t = max(0, (n2 - k) // 2)

    deg_E = t
    deg_N = k - 1 + t
    num_unknowns = (deg_N + 1) + deg_E

    M = gf_zeros(GF, (n2, num_unknowns))
    b = gf_zeros(GF, n2)

    max_deg = max(deg_N, deg_E)
    powers = gf_ones(GF, (max_deg + 1, n2))
    for j in range(1, max_deg + 1):
        powers[j] = powers[j - 1] * A2

    for j in range(deg_N + 1):
        M[:, j] = powers[j]

    b = r2

    for j in range(1, deg_E + 1):
        col = (deg_N + 1) + (j - 1)
        M[:, col] = -(r2 * powers[j])

    Aug = gf_zeros(GF, (n2, num_unknowns + 1))
    Aug[:, :num_unknowns] = M
    Aug[:, -1] = b

    rows, cols_aug = Aug.shape
    cols = cols_aug - 1
    pivot_cols = []
    pivot_count = 0
    row = 0

    for col in range(cols):
        if row >= rows:
            break
        pivot_row = None
        for r in range(row, rows):
            if Aug[r, col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        if pivot_row != row:
            Aug[[row, pivot_row]] = Aug[[pivot_row, row]]
        inv = gf_scalar_one(GF) / Aug[row, col]
        Aug[row, :] *= inv
        if row + 1 < rows:
            below = np.where(Aug[row + 1:, col] != 0)[0]
            if below.size > 0:
                below_rows = below + (row + 1)
                factors = Aug[below_rows, col].reshape(-1, 1)
                Aug[below_rows, :] -= factors * Aug[row, :]
        pivot_cols.append(col)
        pivot_count += 1
        row += 1
        if pivot_count == num_unknowns:
            break

    for r in range(rows):
        if np.all(Aug[r, :cols] == 0) and Aug[r, cols] != 0:
            raise ValueError("Welch decoding failure: inconsistent linear system.")

    x = gf_zeros(GF, num_unknowns)
    for i in range(pivot_count - 1, -1, -1):
        col = pivot_cols[i]
        r = i
        s = Aug[r, cols]
        if col + 1 < cols:
            coeffs = Aug[r, col + 1:cols]
            xs = x[col + 1:cols]
            if xs.size > 0:
                s -= np.sum(coeffs * xs)
        x[col] = s

    N_coeffs = x[:deg_N + 1]
    E_coeffs = gf_zeros(GF, deg_E + 1)
    E_coeffs[0] = gf_scalar_one(GF)
    if deg_E > 0:
        E_coeffs[1:] = x[deg_N + 1:]

    try:
        N_poly = galois.Poly(N_coeffs, field=GF, order="asc")
        E_poly = galois.Poly(E_coeffs, field=GF, order="asc")
        Q, _ = divmod(N_poly, E_poly)
        m_coeffs = Q.coeffs(order="asc")
    except Exception:
        def trim(a):
            i = len(a) - 1
            while i > 0 and a[i] == 0:
                i -= 1
            return a[:i + 1]

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

def Outer_Decode_grs(consensus_output: List[Tuple[int, np.ndarray]], l_out, M, n_workers=None, show_progress=True):
    if not consensus_output:
        return np.empty((0,), dtype=int), np.empty((0, 0), dtype=int)

    ids, payloads = zip(*consensus_output)
    subset_idx = np.fromiter(ids, dtype=int)

    first_len = len(payloads[0])
    if any(len(p) != first_len for p in payloads):
        raise ValueError("All payload vectors must have the same length")

    binary_uhats = np.vstack(payloads).astype(int, copy=False)

    decimal_matrix = []
    for row in binary_uhats:
        blocks = binary_to_decimal_blocks(row.tolist(), l_out)
        decimal_matrix.append(blocks)

    D = np.array(decimal_matrix)
    D_T = D.T

    _ = galois.GF(2**l_out)

    tasks = [(i, D_T[i], M, subset_idx, l_out) for i in range(D_T.shape[0])]
    decoded_rows = [None] * len(tasks)
    n_workers = _cap_workers_for_tasks(n_workers, len(tasks))

    try:
        if n_workers == 1:
            for task in tqdm(tasks, total=len(tasks), desc="Decoding", disable=not show_progress):
                row_idx, message = _decode_row_grs(task)
                decoded_rows[row_idx] = message
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for row_idx, message in tqdm(
                    executor.map(_decode_row_grs, tasks, chunksize=max(1, len(tasks) // (n_workers * 4))),
                    total=len(tasks), desc="Decoding", disable=not show_progress
                ):
                    decoded_rows[row_idx] = message
    except Exception:
        try:
            for task in tqdm(tasks, total=len(tasks), desc="Decoding", disable=not show_progress):
                row_idx, message = _decode_row_grs(task)
                decoded_rows[row_idx] = message
        except Exception:
            return None

    decoded_matrix = np.array(decoded_rows).T

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
    if len(binary_message) % block_length != 0:
        raise ValueError("The length of the binary message must be divisible by the block length.")
    blocks = [binary_message[i:i + block_length] for i in range(0, len(binary_message), block_length)]
    decimal_blocks = [int("".join(map(str, block)), 2) for block in blocks]
    return decimal_blocks

def decimal_to_binary_blocks(decimal_list, block_length):
    return [f"{x:0{block_length}b}" for x in decimal_list]

def _get_cache_key(**kwargs):
    key_json = json.dumps(kwargs, sort_keys=True)
    return hashlib.sha256(key_json.encode("utf-8")).hexdigest()[:16]


def _derive_packet_metadata(params, existing_packet_metadata, outer_rate, k):
    """Return per-packet decode metadata, deriving it for compact JSON files."""
    if existing_packet_metadata:
        return existing_packet_metadata

    original_size_bytes = params.get("original_size_bytes")
    if original_size_bytes is None:
        raise ValueError("No packet metadata found and original_size_bytes is missing")

    total_bits = int(original_size_bytes) * 8
    total_info_strands = math.ceil(total_bits / k) if total_bits else 0
    if total_info_strands <= 0:
        raise ValueError("Cannot decode an empty file with compact packet metadata")

    max_seqs_per_packet = params.get("max_seqs_per_packet", MAX_SEQS_PER_PACKET)
    packet_plan, _ = plan_packetization_by_rate(
        total_info_strands,
        outer_rate,
        max_seqs_per_packet,
    )
    if params.get("filtered"):
        candidate_pool_size = params.get("filtered_candidate_pool_size") or FILTERED_CANDIDATE_POOL_SIZE
        packet_plan, _ = maximize_outer_redundancy_for_filtered_packets(
            packet_plan,
            candidate_pool_size,
        )

    expected_num_packets = params.get("num_packets")
    if expected_num_packets is not None and len(packet_plan) != expected_num_packets:
        raise ValueError(
            "Compact metadata packet plan mismatch: "
            f"derived {len(packet_plan)} packet(s), expected {expected_num_packets}."
        )

    total_pad_size = total_info_strands * k - total_bits
    return [
        {
            "packet_id": packet["packet_id"],
            "M": packet["M"],
            "N_strands": packet["N_strands"],
            "outer_redundancy": packet["outer_redundancy"],
            "pad_size": total_pad_size if idx == len(packet_plan) - 1 else 0,
            **(
                {"selection_required": packet["selection_required"]}
                if "selection_required" in packet else {}
            ),
        }
        for idx, packet in enumerate(packet_plan)
    ]


def _build_packet_id_inverse_map(params, num_packets, seed):
    """Return encoded packet-id -> logical packet-id mapping."""
    packet_id_inverse_map = params.get("packet_id_inverse_map")
    if packet_id_inverse_map is not None:
        return {int(k): v for k, v in packet_id_inverse_map.items()}

    packet_id_map = params.get("packet_id_map")
    if packet_id_map is not None:
        return {v: i for i, v in enumerate(packet_id_map)}

    packet_id_bits = params.get("packet_id_bits", 16)
    rng = np.random.default_rng(seed)
    packet_id_map = rng.permutation(1 << packet_id_bits)[:num_packets]
    return {int(encoded_id): logical_id for logical_id, encoded_id in enumerate(packet_id_map)}


def _decode_single_packet_task(args):
    """Decode one packet end to end from pre-decoded uhats."""
    (
        packet_id,
        packet_uhats,
        metadata,
        l_out,
        filtered,
        seed,
        per_packet_workers,
        outer_rate,
    ) = args

    M = metadata["M"]
    C = rate_to_redundancy(outer_rate, M)
    N_strands = M + C

    if not packet_uhats:
        return packet_id, None, "No DNA sequences found"

    uhat_list = packet_uhats

    if filtered:
        indexed_uhat, _ = consensus_by_strand_id_grs(uhat_list, l_out)
        decoded_packet = Outer_Decode_grs(indexed_uhat, l_out, M, n_workers=per_packet_workers, show_progress=False)
    else:
        if C == 0:
            indexed_uhat = consensus_by_strand_id(uhat_list, l_out, N_strands, seed + packet_id)
            if not indexed_uhat:
                decoded_packet = None
            else:
                sorted_payloads = [payload for idx, payload in sorted(indexed_uhat, key=lambda x: x[0])]
                decoded_packet = np.array(sorted_payloads)
        else:
            indexed_uhat = consensus_by_strand_id(uhat_list, l_out, N_strands, seed + packet_id)
            decoded_binary_matrix = build_outer_decoder_input(indexed_uhat, N_strands)
            decoded_packet = Outer_Decode(decoded_binary_matrix, l_out, C, per_packet_workers, show_progress=False)

    if decoded_packet is None or (hasattr(decoded_packet, 'size') and decoded_packet.size == 0):
        return packet_id, None, "Outer decode failed"

    return packet_id, decoded_packet, None


def process_multipacket_decoding(
    packet_uhat_sequences: Dict[int, List[np.ndarray]],
    packet_metadata: List[dict],
    l_out: int,
    filtered: bool,
    seed: int,
    processes: int,
    *,
    outer_rate: float,
) -> Tuple[List[np.ndarray], List[int]]:
    print(f"\n=== MULTI-PACKET DECODING ===")

    num_packets = len(packet_metadata)
    all_decoded_packets = [None] * num_packets

    packet_workers, per_packet_workers = _plan_multipacket_workers(processes, num_packets)
    print(
        f"Worker plan: {packet_workers} packet worker(s), "
        f"{per_packet_workers} inner/outer worker(s) per packet"
    )

    metadata_by_id = {m["packet_id"]: m for m in packet_metadata}

    tasks = [
        (
            packet_id,
            packet_uhat_sequences.get(packet_id, []),
            metadata_by_id[packet_id],
            l_out,
            filtered,
            seed,
            per_packet_workers,
            outer_rate,
        )
        for packet_id in range(num_packets)
    ]

    def _run_sequential():
        for task in tqdm(tasks, total=len(tasks), desc="Outer decoding"):
            packet_id, decoded_packet, error = _decode_single_packet_task(task)
            if error is not None:
                raise RuntimeError(f"Packet {packet_id} failed: {error}")
            all_decoded_packets[packet_id] = decoded_packet

    def _run_parallel():
        with ProcessPoolExecutor(max_workers=packet_workers) as executor:
            futures = {executor.submit(_decode_single_packet_task, task): task[0] for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Outer decoding"):
                packet_id, decoded_packet, error = future.result()
                if error is not None:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError(f"Packet {packet_id} failed: {error}")
                all_decoded_packets[packet_id] = decoded_packet

    try:
        if packet_workers <= 1:
            _run_sequential()
        else:
            try:
                _run_parallel()
            except RuntimeError:
                raise
            except Exception:
                _run_sequential()
    except RuntimeError as e:
        print(f"\n❌ Decoding terminated: {e}")
        return all_decoded_packets, []

    return all_decoded_packets, []


def decode(
    file_name,
    input_path=None,
    processes=None,
    return_details=False,
):
    """
    Decode DNA file from a TXT file with multi-packet support.

    For multi-packet files:
    - DNA sequences are grouped by packet ID (extracted from barcode)
    - Each packet is decoded independently
    - Decoded data is concatenated in packet order

    For single-packet files, behavior is identical to the original.

    Args:
        file_name (str): Name of the text file to decode (e.g., "encoded_file.txt").
        input_path (str or Path, optional): Folder where the file is located.
                                            Defaults to current working directory.
        processes (int, optional): Total process budget for parallel decoding.
        return_details (bool): Return a diagnostics dictionary instead of only success.
    
    Note:
        The random seed is automatically read from encoding_params.json.
        No manual seed input is required.
    """
    inner_decoding_time = 0.0
    total_decoding_time = 0.0
    dmin = 5
    marker_period = 2
    l_in = 8
    l_out = 16
    codebook = load_codebook_dna()

    encoded_json_filename = "encoding_params.json"
    meta_path = os.path.join(os.getcwd(), encoded_json_filename)

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
    outer_rate = params.get("outer_rate")
    filtered = params["filtered"]
    useMarker = params["useMarker"]
    original_extension = params["original_extension"]
    seed = params.get("seed")
    
    if seed is None:
        raise ValueError(
            "Random seed not found in encoding parameters. "
            "Please ensure you're using a valid encoding_params.json file from the new codec version."
        )

    if outer_rate is None:
        if "outer_redundancy" in params:
            raise ValueError(
                "Old encoding format detected (uses outer_redundancy instead of outer_rate). "
                "Please re-encode your file with the new codec version."
            )
        raise ValueError("outer_rate not found in encoding parameters")

    is_multipacket = params.get("is_multipacket", False)
    num_packets = params.get("num_packets", 1)
    packet_id_bits = params.get("packet_id_bits", 16 if is_multipacket else 0)
    packet_metadata = _derive_packet_metadata(
        params,
        metadata.get("packets", []),
        outer_rate,
        k,
    )

    if not is_multipacket:
        if packet_metadata:
            M_single = packet_metadata[0]["M"]
            C_single = rate_to_redundancy(outer_rate, M_single)
            N_strands = M_single + C_single
            pad_size = packet_metadata[0].get("pad_size", 0)
            C = C_single
        else:
            raise ValueError("No packet metadata found for single-packet decoding")

    processes = _normalize_worker_count(processes)
    print(f"Using process budget: {processes}")

    if input_path is None:
        input_path = Path.cwd()
    else:
        input_path = Path(input_path)

    file_path = input_path / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"File to decode not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        consensuses = [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]

    print(f"Loaded {len(consensuses)} sequences from {file_path}")

    start_time = time.perf_counter()
    per_strand_profile_summary = None
    profile_mode = "estimated" if c1 != 0 and useMarker else "not_applicable"

    if is_multipacket:
        print("Inner decoding all DNA sequences...")
        if c1 != 0:
            if useMarker:
                inner_decoding_time, uhats, per_strand_profile_summary = parallel_decode_realtime(
                    consensuses,
                    n, l_in, N, K, c1, marker_period, q,
                    n_workers=processes,
                )
            else:
                len_last = (k - 1) % l_in + 1
                lim = 6
                lambda_depths = [0] * lim
                lambda_depths[0] = 1

                cache_dir = os.path.join(os.path.expanduser("~"), ".mgcp_cache")
                os.makedirs(cache_dir, exist_ok=True)
                cache_key = _get_cache_key(
                    lambda_depths=lambda_depths, K=K, len_last=len_last, lim=lim, c1=c1,
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
                    Patterns, codebook, dmin, opt=False, n_workers=processes
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

        print("Grouping DNA sequences by packet ID...")
        packet_sequences = {}
        inv_map = _build_packet_id_inverse_map(params, num_packets, seed)

        for uhat in uhats:
            if uhat is None or len(uhat) < packet_id_bits + l_out:
                continue
            packet_id_bits_extracted = uhat[:packet_id_bits]
            packet_id_decimal = 0
            for bit in packet_id_bits_extracted:
                packet_id_decimal = (packet_id_decimal << 1) | int(bit)

            packet_id = inv_map.get(packet_id_decimal, -1)
            if packet_id == -1:
                continue

            payload_with_strand_id = uhat[packet_id_bits:]
            packet_sequences.setdefault(packet_id, []).append(payload_with_strand_id)

        total_grouped = sum(len(v) for v in packet_sequences.values())
        print(f"  Grouped {total_grouped} DNA sequences into {len(packet_sequences)} packet buckets")

        print("Outer decoding...")

        all_decoded_packets, failed_packets = process_multipacket_decoding(
            packet_sequences,
            packet_metadata,
            l_out,
            filtered,
            seed,
            processes,
            outer_rate=outer_rate,
        )

        if failed_packets:
            print(f"\nWARNING: Failed to decode packets: {failed_packets}")

        successful_packets = [p for p in all_decoded_packets if p is not None and p.size > 0]
        if len(successful_packets) == len(packet_metadata):
            decoded_file = np.vstack(successful_packets)
            success = True
            total_pad_size = sum(packet["pad_size"] for packet in packet_metadata)
        else:
            success = False
            decoded_file = None

    else:
        print("Inner decoding...")
        if c1 != 0:
            if useMarker:
                inner_decoding_time, uhats, per_strand_profile_summary = parallel_decode_realtime(
                    consensuses,
                    n, l_in, N, K, c1, marker_period, q,
                    n_workers=processes,
                )
            else:
                len_last = (k - 1) % l_in + 1
                lim = 6
                lambda_depths = [0] * lim

                cache_dir = os.path.join(os.path.expanduser("~"), ".mgcp_cache")
                os.makedirs(cache_dir, exist_ok=True)
                cache_key = _get_cache_key(
                    lambda_depths=lambda_depths, K=K, len_last=len_last, lim=lim, c1=c1,
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
                    Patterns, codebook, dmin, opt=False, n_workers=processes
                )
        else:
            inner_start_time = time.perf_counter()
            uhats = []
            for consensus_seq in consensuses:
                if len(consensus_seq) == n:
                    uhat = dna_to_binary(consensus_seq)
                else:
                    uhat = None
                uhats.append(uhat)
            inner_decoding_time = time.perf_counter() - inner_start_time

        uhat_list = [uhat for uhat in uhats if uhat is not None]

        print("Outer decoding...")

        if filtered:
            indexed_uhat, strand_to_positions = consensus_by_strand_id_grs(uhat_list, l_out)
            filter_file_path = "filter_indices.txt"
            if os.path.exists(filter_file_path):
                with open(filter_file_path, 'r') as f:
                    valid_indices = set(int(line.strip()) for line in f if line.strip())
                indexed_uhat_filtered = [(idx, payload) for idx, payload in indexed_uhat if idx in valid_indices]
            else:
                indexed_uhat_filtered = indexed_uhat
            decoded_file = Outer_Decode_grs(indexed_uhat_filtered, l_out, N_strands - C, n_workers=processes)
        else:
            indexed_uhat = consensus_by_strand_id(uhat_list, l_out, N_strands, seed)
            decoded_binary_matrix = build_outer_decoder_input(indexed_uhat, N_strands)
            decoded_file = Outer_Decode(decoded_binary_matrix, l_out, C, processes)

        success = decoded_file is not None
        total_pad_size = pad_size

    total_decoding_time = time.perf_counter() - start_time

    if success and decoded_file is not None:
        flat_bits = np.concatenate(decoded_file).astype(np.uint8)
        if total_pad_size > 0:
            flat_bits = flat_bits[:-total_pad_size]
        decoded_bytes = np.packbits(flat_bits)

        decoded_filename = f"decoded_output{original_extension}"
        decoded_path = os.path.join(os.getcwd(), decoded_filename)
        with open(decoded_path, "wb") as f:
            f.write(decoded_bytes.tobytes())

        print(f"\n✅ Decoded file saved to {decoded_path}")
        print(f"Total decoding time: {total_decoding_time:.2f} s")
    else:
        decoded_path = None
        print("\n❌ Decoding failed — No file saved")

    if return_details:
        return {
            "success": bool(success),
            "profile_mode": profile_mode,
            "Pd": None,
            "Pi": None,
            "Ps": None,
            "Pe": None,
            "calibration": None,
            "calibration_time_sec": 0.0,
            "per_strand_profile_summary": per_strand_profile_summary,
            "inner_decoding_time_sec": inner_decoding_time,
            "total_decoding_time_sec": total_decoding_time,
            "decoded_path": decoded_path,
            "sequence_count": len(consensuses),
        }
    return success
