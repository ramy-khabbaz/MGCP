import os
import numpy as np
from reedsolo import RSCodec
import math
import galois
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from mgcp.dna.MGCP_Encode_DNA_p2 import MGCP_Encode_DNA_p2
from mgcp.dna.GCP_Encode_DNA import GCP_Encode_DNA_brute
from mgcp.utils.loader import load_codebook_dna
from mgcp.dna.codec.multipacket_utils import (
    split_file_into_packets,
    FILTERED_CANDIDATE_POOL_SIZE,
    plan_packetization_by_rate,
    maximize_outer_redundancy_for_filtered_packets,
)

def binary_to_dna(binary_message):
    dna_map = {'00': 'A', '01': 'T', '10': 'C', '11': 'G'}
    return "".join(dna_map["".join(map(str, binary_message[i:i+2]))] for i in range(0, len(binary_message), 2))

def binary_to_decimal_blocks(binary_message, block_length):
    if len(binary_message) % block_length != 0:
        raise ValueError("The length of the binary message must be divisible by the block length.")
    blocks = [binary_message[i:i + block_length] for i in range(0, len(binary_message), block_length)]
    decimal_blocks = [int("".join(map(str, block)), 2) for block in blocks]
    return decimal_blocks

def decimal_to_binary_blocks(decimal_list, block_length):
    return [f"{x:0{block_length}b}" for x in decimal_list]

def Outer_Encode(u, l, C):
    M, k = u.shape
    if k % l != 0:
        raise ValueError("Each row length must be divisible by the block length.")
    num_blocks = k // l
    D = np.zeros((M, num_blocks), dtype=int)
    for i in range(M):
        D[i, :] = binary_to_decimal_blocks(u[i, :].tolist(), l)
    D_transposed = D.T
    rsEncoder = RSCodec(C, c_exp=l)
    encoded_rows = []
    for row in D_transposed:
        encoded_row = list(rsEncoder.encode(list(row)))
        encoded_rows.append(encoded_row)
    encoded_matrix = np.array(encoded_rows)
    encoded_matrix_T = encoded_matrix.T
    final_binary_matrix = []
    for row in encoded_matrix_T:
        binary_strings = decimal_to_binary_blocks(row.tolist(), l)
        binary_row = [int(bit) for bin_str in binary_strings for bit in bin_str]
        final_binary_matrix.append(binary_row)
    final_binary_matrix = np.array(final_binary_matrix)
    return final_binary_matrix

def int_to_bits(n: int, width: int):
    """MSB-first binary of n with fixed width (list of 0/1 ints)."""
    if n < 0 or n >= (1 << width):
        raise ValueError(f"int_to_bits: {n=} doesn't fit in width={width}")
    return [(n >> (width - 1 - i)) & 1 for i in range(width)]

def _encode_packet_task(args):
    """Encode a single packet for parallel execution."""
    (
        packet_id,
        packet_cfg,
        packet_bits,
        M_packet,
        is_multipacket,
        packet_id_bits,
        k,
        l_out,
        l_in,
        inner_redundancy,
        filtered,
        seed,
        codebook,
        subset_idx,
        useMarker,
    ) = args

    total_needed = M_packet * k
    pad_size = total_needed - packet_bits.size
    if pad_size > 0:
        rng = np.random.default_rng(seed + packet_id)
        pad = rng.integers(0, 2, size=pad_size, dtype=np.uint8)
        packet_bits = np.concatenate([packet_bits, pad])

    u = packet_bits.reshape(M_packet, k)
    C = packet_cfg["outer_redundancy"]
    if filtered:
        u_outer = Outer_Encode_grs(u, l_out, subset_idx)
    else:
        u_outer = Outer_Encode(u, l_out, C)
    N_strands = len(u_outer)

    rng = np.random.default_rng(seed + packet_id)
    idx_map = rng.permutation(2**l_out)[:N_strands]
    fwd_map = {i: idx_map[i] for i in range(N_strands)}

    encoded_strands_packet = []
    n = None
    N = None
    K = None
    q = None

    if inner_redundancy == 0:
        for idx, row in enumerate(u_outer):
            if filtered:
                idx_bits = int_to_bits(subset_idx[idx], l_out)
            else:
                idx_bits = int_to_bits(fwd_map[idx], l_out)
            if is_multipacket:
                strand_prefix = np.array(packet_id_bits + idx_bits, dtype=row.dtype)
            else:
                strand_prefix = np.array(idx_bits, dtype=row.dtype)
            payload_dna = binary_to_dna(row)
            strand_id_dna = binary_to_dna(strand_prefix)
            encoded_strands_packet.append(strand_id_dna + payload_dna)
        n = len(strand_id_dna + payload_dna) if encoded_strands_packet else 0
    else:
        for idx, row in enumerate(u_outer):
            if filtered:
                idx_bits = int_to_bits(subset_idx[idx], l_out)
            else:
                idx_bits = int_to_bits(fwd_map[idx], l_out)
            if is_multipacket:
                strand_prefix = np.array(packet_id_bits + idx_bits, dtype=row.dtype)
            else:
                strand_prefix = np.array(idx_bits, dtype=row.dtype)
            row_with_prefix = np.concatenate([strand_prefix, row])

            if useMarker:
                x, n, N, K, q, _, _ = MGCP_Encode_DNA_p2(row_with_prefix, l_in, inner_redundancy, codebook)
            else:
                x, n, N, K, q, _, _ = GCP_Encode_DNA_brute(row_with_prefix, l_in, inner_redundancy, codebook)

            encoded_strands_packet.append(x)

    return packet_id, encoded_strands_packet, n, N, K, q

def gf_zeros(GF, n):
    return GF.Zeros(n) if hasattr(GF, "Zeros") else GF(np.zeros(n, dtype=int))

def gf_ones(GF, n):
    return GF.Ones(n) if hasattr(GF, "Ones") else GF(np.ones(n, dtype=int))

def grs_encode_subset(u, c_exp=16, subset_idx=None, fcr=0, v=None, prim=None):
    if prim is None:
        GF = galois.GF(2**c_exp)
    else:
        GF = galois.GF(2**c_exp, irreducible_poly=prim)
    u = GF(u)
    n = 2**c_exp
    if subset_idx is None:
        subset_idx = np.arange(n, dtype=int)
    else:
        subset_idx = np.asarray(subset_idx, dtype=int)
        if np.any((subset_idx < 0) | (subset_idx >= n)):
            raise ValueError("subset_idx out of range")
    alpha = GF.primitive_element
    locators_all = np.concatenate([
        alpha ** (fcr + np.arange(n - 1, dtype=np.int64)),
        gf_zeros(GF, 1),
    ])
    A = locators_all[subset_idx]
    if v is None:
        v = gf_ones(GF, A.size)
    else:
        v = GF(v)
        if v.size != A.size:
            raise ValueError("v length must equal len(subset_idx)")
    y = gf_zeros(GF, A.size)
    power = gf_ones(GF, A.size)
    for coeff in u:
        y += coeff * power
        power *= A
    y *= v
    return y

def Outer_Encode_grs(u, l, subset_idx=None, fcr=0, prim=None, C=None):
    M, k = u.shape
    if k % l != 0:
        raise ValueError("Each row length must be divisible by the block length.")
    num_blocks = k // l
    if subset_idx is None:
        if C is None:
            C = 2**l - M
        n_short = M + C
        subset_idx = np.arange(n_short, dtype=int)
    else:
        subset_idx = np.asarray(subset_idx, dtype=int)
        n_parent = 2**l
        if np.any((subset_idx < 0) | (subset_idx >= n_parent)):
            raise ValueError("subset_idx out of range")
        n_short = subset_idx.size
    D = np.zeros((M, num_blocks), dtype=int)
    for i in range(M):
        D[i, :] = binary_to_decimal_blocks(u[i, :].tolist(), l)
    D_transposed = D.T
    encoded_rows = []
    for row in D_transposed:
        y = grs_encode_subset(row, c_exp=l, subset_idx=subset_idx, fcr=fcr)
        encoded_rows.append([int(el) for el in y])
    encoded_matrix = np.array(encoded_rows)
    encoded_matrix_T = encoded_matrix.T
    final_binary_matrix = []
    for row in encoded_matrix_T:
        bin_strs = decimal_to_binary_blocks(row.tolist(), l)
        bits = [int(b) for s in bin_strs for b in s]
        final_binary_matrix.append(bits)
    final_binary_matrix = np.array(final_binary_matrix)
    return final_binary_matrix

def calculate_best_k(max_length, c1, useMarker, l_in=8, l_out=16):
    """
    Finds the best k (multiple of l_out) for a target oligo length.
    Returns (best_k, estimated_length).
    """
    if l_in > 8:
        raise ValueError(f"Invalid l_in={l_in}. It must not exceed 8.")

    def calc_length(k, c1):
        if c1 != 0:
            if useMarker:
                return (16 + 4 + k + ((k * 4) / (2 * l_in)) + c1 * l_in + (c1 / 2) * 4 + 24) / 2
            else:
                return (16 + k + 24 + c1 * l_in) / 2
        else:
            return (16 + k) / 2

    best_k = None
    best_diff = float("inf")
    best_len = None

    for candidate_k in range(l_out, 20000, l_out):
        num_blocks = candidate_k // l_in
        if useMarker and c1 != 0:
            total = num_blocks + c1
            if total % 2 != 0:
                continue
        current_len = calc_length(candidate_k, c1)
        if current_len > max_length:
            continue
        diff = abs(current_len - max_length)
        if diff < best_diff:
            best_diff = diff
            best_k = candidate_k
            best_len = current_len
        if diff < 0.5:
            break

    if best_k is None:
        raise ValueError(
            f"No valid encoding setting found satisfying conditions for max_length={max_length}"
        )

    return best_k, int(best_len)

def _get_actual_oligo_length(k, l_in, l_out, inner_redundancy, useMarker, is_multipacket, codebook):
    """
    Test-encode one example DNA sequence to get the ACTUAL oligo length (not estimated).
    """
    if is_multipacket:
        test_input = np.zeros(16 + l_out + k, dtype=int)
    else:
        test_input = np.zeros(l_out + k, dtype=int)

    if inner_redundancy == 0:
        dna = binary_to_dna(test_input)
        return len(dna)
    else:
        if useMarker:
            _, actual_n, _, _, _, _, _ = MGCP_Encode_DNA_p2(test_input, l_in, inner_redundancy, codebook)
        else:
            _, actual_n, _, _, _, _, _ = GCP_Encode_DNA_brute(test_input, l_in, inner_redundancy, codebook)
        return actual_n


def _resolve_k_and_multipacket(bits_size, max_length, inner_redundancy, useMarker,
                                outer_rate, max_seqs_per_packet, l_in, l_out):
    """
    Determine k and is_multipacket in a single consistent pass (fixes bug #1).

    The old code had a circular dependency: it used k_single to decide if multipacket
    was needed, then recalculated k for multipacket (smaller, due to reserved NTs),
    then re-planned with the new k — but never re-checked whether the new k actually
    still required multipacket. This meant is_multipacket_planned (used to compute
    actual_n) could disagree with is_multipacket (used for actual encoding), causing
    the wrong oligo length n to be written to encoding_params.json and breaking decode.

    Strategy:
      1. Probe with k_multi (conservative: reserves 8 NTs for packet ID).
      2. Plan with k_multi → get definitive is_multipacket.
      3. If single-packet, upgrade to k_single (larger, no reservation needed)
         and verify it still fits in one packet.
      4. Return a fully consistent (k, is_multipacket) pair.
    """
    reserved_nt = 8
    effective_max_length = max_length - reserved_nt
    if effective_max_length < 20:
        raise ValueError(
            f"max_length={max_length} is too small for multipacket encoding "
            f"(need at least 28; {reserved_nt} NTs are reserved for the packet ID)"
        )
    k_multi, _ = calculate_best_k(effective_max_length, inner_redundancy, useMarker, l_in, l_out)

    M_multi = math.ceil(bits_size / k_multi)
    packet_plan, _ = plan_packetization_by_rate(M_multi, outer_rate, max_seqs_per_packet)
    is_multipacket = len(packet_plan) > 1

    if is_multipacket:
        k = k_multi
    else:
        k_single, _ = calculate_best_k(max_length, inner_redundancy, useMarker, l_in, l_out)
        M_single = math.ceil(bits_size / k_single)
        packet_plan_single, _ = plan_packetization_by_rate(M_single, outer_rate, max_seqs_per_packet)
        if len(packet_plan_single) == 1:
            k = k_single
        else:
            k = k_multi
            is_multipacket = True

    return k, is_multipacket

def encode(file_name, max_length, inner_redundancy, outer_rate, input_path=None,
           useMarker=True, filtered=False, seed=123456, max_seqs_per_packet=None,
           processes=None):
    
    """
    Encode a binary file into DNA sequences using DNA-MGC+ codec with multi-packet support.

    For files requiring multiple packets (>max_seqs_per_packet DNA sequences per packet):
    - File is split into packets, each max max_seqs_per_packet DNA sequences
    - Outer redundancy is calculated per packet based on outer_rate
    - Each DNA sequence gets a 16-bit packet ID prepended to the binary data before inner encoding
    - Packet IDs are randomized using pseudorandom permutation

    Filtered mode keeps the normal packet plan, but emits all 2^16 addressable
    candidate DNA sequences for every packet. The requested outer_rate is used
    to report how many candidates the user should retain from each packet.

    For single-packet files, behavior is identical to original (no packet overhead).

    Args:
        file_name (str): Name of the input file to encode (e.g., "data.bin").
        input_path (str or Path, optional): Folder where the file is located. Defaults to cwd.
        max_length (int): Max length of each oligo.
        inner_redundancy (int): Inner redundancy parameter (same for all packets).
        outer_rate (float): Outer code rate (e.g., 0.9 means 90% of sequences are
                           information sequences, 10% are redundancy). Must be in (0, 1).
        useMarker (bool): Use marker-based encoding.
        filtered (bool): Whether to emit the full 2^16 candidate pool per packet
                         for manual filtering.
        seed (int): Random seed for reproducibility.
        max_seqs_per_packet (int, optional): Maximum DNA sequences per packet after outer encoding.
                                                Defaults to 2^12 (4096).
        processes (int, optional): Number of worker processes to parallelize packet encoding.
                                   Defaults to CPU count.
    """
        
    if processes is None:
        processes = max(1, os.cpu_count() or 1)

    l_in = 8
    l_out = 16
    if max_seqs_per_packet is None:
        max_seqs_per_packet = 2**12
    MAX_SEQS = 1 << l_out

    if not (0 < outer_rate < 1):
        raise ValueError(f"outer_rate must be in (0, 1), got {outer_rate}")

    if input_path is None:
        input_path = Path.cwd()
    else:
        input_path = Path(input_path)

    input_file_path = input_path / file_name
    if not input_file_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
    file_extension = input_file_path.suffix

    print(f"Encoding file: {input_file_path}")

    codebook = load_codebook_dna()
    packet_id_bits_length = 16

    data_bytes = input_file_path.read_bytes()
    byte_arr = np.frombuffer(data_bytes, dtype=np.uint8)
    bits = np.unpackbits(byte_arr)

    k, is_multipacket = _resolve_k_and_multipacket(
        bits_size=bits.size,
        max_length=max_length,
        inner_redundancy=inner_redundancy,
        useMarker=useMarker,
        outer_rate=outer_rate,
        max_seqs_per_packet=max_seqs_per_packet,
        l_in=l_in,
        l_out=l_out,
    )

    actual_n = _get_actual_oligo_length(k, l_in, l_out, inner_redundancy, useMarker, is_multipacket, codebook)
    print(f"Information bits length k = {k}b, oligo length = {actual_n} NTs")

    M = math.ceil(bits.size / k)
    packet_plan, total_outer_redundancy = plan_packetization_by_rate(M, outer_rate, max_seqs_per_packet)
    num_packets_needed = len(packet_plan)

    assert (num_packets_needed > 1) == is_multipacket, (
        f"Packet count mismatch: is_multipacket={is_multipacket} but "
        f"plan_packetization returned {num_packets_needed} packets. "
        f"This is a bug in _resolve_k_and_multipacket."
    )

    if num_packets_needed > MAX_SEQS:
        raise ValueError(
            f"Need {num_packets_needed} packets, but only {MAX_SEQS} sequence IDs are available with l_out={l_out}."
        )

    if filtered:
        packet_plan, total_outer_redundancy = maximize_outer_redundancy_for_filtered_packets(
            packet_plan,
            FILTERED_CANDIDATE_POOL_SIZE,
        )

    packets = split_file_into_packets(bits, k, [packet["M"] for packet in packet_plan])

    if is_multipacket:
        packet_id_rng = np.random.default_rng(seed)
        packet_id_map = packet_id_rng.permutation(2**16)[:num_packets_needed].tolist()
    else:
        packet_id_map = None

    print(f"\n=== {'MULTI' if is_multipacket else 'SINGLE'}-PACKET ENCODING ===")
    print(f"Number of packets: {num_packets_needed}")
    if filtered:
        print(f"Filtered candidate pool: {FILTERED_CANDIDATE_POOL_SIZE} DNA sequences per packet")
        print(f"Requested outer rate: {outer_rate}")
        for packet in packet_plan:
            required = packet["selection_required"]
            achieved_rate = packet["M"] / required
            print(
                f"Packet {packet['packet_id'] + 1}: choose {required} of "
                f"{FILTERED_CANDIDATE_POOL_SIZE} sequences "
                f"(M={packet['M']}, resulting rate={achieved_rate:.6f})"
            )
    # print(
    #     "Packet layout (M information sequences, C outer redundancy, N post-outer DNA sequences): "
    #     + ", ".join(
    #         f"p{packet['packet_id']}=(M={packet['M']}, C={packet['outer_redundancy']}, N={packet['N_strands']})"
    #         for packet in packet_plan
    #     )
    # )

    packet_tasks = []
    total_n = None
    N_global = None
    K_global = None
    q_global = None

    subset_idx_list = []
    if filtered:
        full_candidate_indices = np.arange(FILTERED_CANDIDATE_POOL_SIZE, dtype=int)
        subset_idx_list = [full_candidate_indices] * num_packets_needed
    else:
        subset_idx_list = [None] * num_packets_needed

    for packet_cfg, (packet_bits, M_packet), subset_idx in zip(packet_plan, packets, subset_idx_list):
        packet_id = packet_cfg["packet_id"]
        if is_multipacket:
            pkt_id_bits = int_to_bits(packet_id_map[packet_id], packet_id_bits_length)
        else:
            pkt_id_bits = None
        packet_tasks.append(
            (
                packet_id,
                packet_cfg,
                packet_bits,
                M_packet,
                is_multipacket,
                pkt_id_bits,
                k,
                l_out,
                l_in,
                inner_redundancy,
                filtered,
                seed,
                codebook,
                subset_idx,
                useMarker,
            )
        )

    all_encoded_strands_by_id = {}

    if processes > 1 and num_packets_needed > 1:
        with ProcessPoolExecutor(max_workers=processes) as executor:
            future_to_packet = {
                executor.submit(_encode_packet_task, task): task[0]
                for task in packet_tasks
            }
            for future in tqdm(as_completed(future_to_packet), total=num_packets_needed, desc="Encoding"):
                packet_id, encoded_strands_packet, n, N, K, q = future.result()
                all_encoded_strands_by_id[packet_id] = encoded_strands_packet
                total_n = n
                N_global = N
                K_global = K
                q_global = q
    else:
        for task in tqdm(packet_tasks, total=num_packets_needed, desc="Encoding"):
            packet_id, encoded_strands_packet, n, N, K, q = _encode_packet_task(task)
            all_encoded_strands_by_id[packet_id] = encoded_strands_packet
            total_n = n
            N_global = N
            K_global = K
            q_global = q

    all_encoded_strands = []
    for pid in range(num_packets_needed):
        all_encoded_strands.extend(all_encoded_strands_by_id[pid])

    output_dict = {
        "params": {
            "num_packets": num_packets_needed,
            "is_multipacket": is_multipacket,
            "original_size_bytes": len(data_bytes),
            "max_seqs_per_packet": max_seqs_per_packet,
            "seed": seed,
            "k": k,
            "n": int(total_n) if total_n is not None else None,
            "N": int(N_global) if N_global is not None else None,
            "K": int(K_global) if K_global is not None else None,
            "q": q_global,
            "l_out": l_out,
            "l_in": l_in,
            "inner_redundancy": inner_redundancy,
            "outer_rate": float(outer_rate),
            "original_extension": file_extension,
            "useMarker": useMarker,
            "filtered": filtered,
            "filtered_candidate_pool_size": FILTERED_CANDIDATE_POOL_SIZE if filtered else None,
            "packet_id_bits": 16 if is_multipacket else 0,
        },
    }

    txt_filename = "encoded_file.txt"
    json_filename = "encoding_params.json"
    txt_path = os.path.join(os.getcwd(), txt_filename)
    json_path = os.path.join(os.getcwd(), json_filename)

    with open(txt_path, "w", encoding="utf-8") as f:
        for pid in range(num_packets_needed):
            if filtered:
                f.write(
                    f"# PACKET {pid + 1}/{num_packets_needed} "
                    f"(choose {packet_plan[pid]['selection_required']} of "
                    f"{FILTERED_CANDIDATE_POOL_SIZE})\n"
                )
            for sequence in all_encoded_strands_by_id[pid]:
                f.write(f"{sequence}\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, indent=2)

    print(f"\nEncoded file saved to {txt_path}")
    print(f"Encoding parameters saved to {json_path}")
    print(f"Total DNA sequences: {len(all_encoded_strands)}")

    return np.array(all_encoded_strands)
