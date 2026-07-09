"""
Multi-packet utilities for DNA codec.

This module provides functionality for:
  - Planning packet sizes with the configured DNA sequence limit enforced after outer encoding
  - Splitting large files into packets
  - Managing packet IDs for multi-packet encoding
"""

import numpy as np
from typing import List, Tuple, Dict
import math


MAX_SEQS_PER_PACKET = 2**12  # 4096 DNA sequences per packet after outer encoding
PACKET_ID_BITS = 16
FILTERED_CANDIDATE_POOL_SIZE = 2**16


def rate_to_redundancy(outer_rate: float, M: int) -> int:
    """
    Convert outer rate to number of redundancy DNA sequences for a given M.
    
    outer_rate = M / (M + C), so C = M * (1 - outer_rate) / outer_rate
    
    Args:
        outer_rate (float): Outer code rate (e.g., 0.9 means 90% of sequences
                            are information sequences, 10% are redundancy)
        M (int): Number of information sequences
    
    Returns:
        int: Number of redundancy DNA sequences C
    
    Raises:
        ValueError: If outer_rate is not in (0, 1) or M <= 0
    """
    if not (0 < outer_rate < 1):
        raise ValueError(f"outer_rate must be in (0, 1), got {outer_rate}")
    if M <= 0:
        raise ValueError(f"M must be > 0, got {M}")
    
    C = M * (1 - outer_rate) / outer_rate
    return int(round(C))


def calculate_total_redundancy_from_rate(
    total_info_strands: int,
    outer_rate: float,
    max_encoded_strands_per_packet: int = MAX_SEQS_PER_PACKET,
) -> int:
    """
    Calculate total redundancy needed from outer_rate and total information sequences.
    
    This finds the minimum number of packets needed, then calculates total
    redundancy such that each packet gets a fairly distributed per-packet rate.
    
    Args:
        total_info_strands (int): Total information sequences
        outer_rate (float): Outer code rate (e.g., 0.9 means 90% of sequences
                            are information sequences, 10% are redundancy)
        max_encoded_strands_per_packet (int): Max DNA sequences per packet after outer encoding
    
    Returns:
        int: Total redundancy DNA sequences needed
    """
    if not (0 < outer_rate < 1):
        raise ValueError(f"outer_rate must be in (0, 1), got {outer_rate}")
    if total_info_strands <= 0:
        raise ValueError(f"total_info_strands must be > 0, got {total_info_strands}")
    
    estimated_total_C = total_info_strands * (1 - outer_rate) / outer_rate
    estimated_total_encoded = total_info_strands + estimated_total_C
    num_packets = max(1, (int(estimated_total_encoded) + max_encoded_strands_per_packet - 1) // max_encoded_strands_per_packet)
    
    info_per_packet = _distribute_evenly(total_info_strands, num_packets)
    
    C_per_packet = [rate_to_redundancy(outer_rate, M_i) for M_i in info_per_packet]
    total_C = sum(C_per_packet)
    
    return total_C


def allocate_redundancy_fairly(total_redundancy: int, num_packets: int) -> List[int]:
    """
    Allocate outer redundancy fairly across packets.
    
    The base redundancy is distributed evenly, with remainder distributed
    to earlier packets for fairness and logical balance.
    
    Args:
        total_redundancy (int): Total outer redundancy specified by user
        num_packets (int): Total number of packets
    
    Returns:
        List[int]: Redundancy per packet (length = num_packets)
    
    Example:
        >>> allocate_redundancy_fairly(10, 3)
        [4, 3, 3]  # First packet gets extra 1
        
        >>> allocate_redundancy_fairly(10, 4)
        [3, 3, 2, 2]
    """
    if num_packets <= 0:
        raise ValueError("num_packets must be > 0")
    if total_redundancy < 0:
        raise ValueError("total_redundancy must be >= 0")
    
    base_redundancy = total_redundancy // num_packets
    remainder = total_redundancy % num_packets
    
    redundancy_per_packet = []
    for i in range(num_packets):
        if i < remainder:
            redundancy_per_packet.append(base_redundancy + 1)
        else:
            redundancy_per_packet.append(base_redundancy)
    
    return redundancy_per_packet


def _distribute_evenly(total: int, buckets: int) -> List[int]:
    """Split an integer total into near-equal non-negative bucket sizes."""
    if buckets <= 0:
        raise ValueError("buckets must be > 0")
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if idx < remainder else 0) for idx in range(buckets)]


def plan_packetization(
    total_info_strands: int,
    total_outer_redundancy: int,
    max_encoded_strands_per_packet: int = MAX_SEQS_PER_PACKET,
) -> List[dict]:
    """
    Plan packet sizes while enforcing the DNA sequence limit after outer encoding.

    The key invariant is:
        M_packet + C_packet <= max_encoded_strands_per_packet

    We find the minimum number of packets such that the worst-case packet size
    ceil(M/n) + ceil(C/n) <= max_encoded_strands_per_packet, then distribute
    both information sequences and redundancy as evenly as possible across packets.

    The old approach computed num_packets from the average packet size, which
    could cause individual packets to overflow by up to 2 DNA sequences because
    _distribute_evenly adds a remainder of +1 to both M and C on the same
    early packets independently.
    """
    if total_info_strands <= 0:
        raise ValueError("total_info_strands must be > 0")
    if total_outer_redundancy < 0:
        raise ValueError("total_outer_redundancy must be >= 0")
    if max_encoded_strands_per_packet <= 0:
        raise ValueError("max_encoded_strands_per_packet must be > 0")

    total_encoded_strands = total_info_strands + total_outer_redundancy
    num_packets = max(1, (total_encoded_strands + max_encoded_strands_per_packet - 1) // max_encoded_strands_per_packet)

    while (
        math.ceil(total_info_strands / num_packets) +
        math.ceil(total_outer_redundancy / num_packets)
        > max_encoded_strands_per_packet
    ):
        num_packets += 1

    info_per_packet = _distribute_evenly(total_info_strands, num_packets)
    redundancy_per_packet = _distribute_evenly(total_outer_redundancy, num_packets)

    packet_plan = []
    for packet_id, (info_count, redundancy_count) in enumerate(zip(info_per_packet, redundancy_per_packet)):
        encoded_count = info_count + redundancy_count
        if encoded_count > max_encoded_strands_per_packet:
            raise ValueError(
                "Packet planning exceeded the post-encoding DNA sequence limit: "
                f"packet {packet_id} needs {encoded_count} DNA sequences, limit is {max_encoded_strands_per_packet}."
            )
        packet_plan.append(
            {
                "packet_id": packet_id,
                "M": info_count,
                "outer_redundancy": redundancy_count,
                "N_strands": encoded_count,
            }
        )

    return packet_plan


def plan_packetization_by_rate(
    total_info_strands: int,
    outer_rate: float,
    max_encoded_strands_per_packet: int = MAX_SEQS_PER_PACKET,
) -> Tuple[List[dict], int]:
    """
    Plan packet sizes and calculate total redundancy based on outer_rate.

    For each packet, C_i = round(M_i * (1 - outer_rate) / outer_rate)
    This ensures each packet gets the target rate, with fair distribution of M.

    Args:
        total_info_strands (int): Total information sequences
        outer_rate (float): Outer code rate (e.g., 0.9 means 90% of sequences
                            are information sequences, 10% are redundancy)
        max_encoded_strands_per_packet (int): Max DNA sequences per packet after outer encoding

    Returns:
        Tuple[List[dict], int]: (packet_plan, total_redundancy)
    """
    if not (0 < outer_rate < 1):
        raise ValueError(f"outer_rate must be in (0, 1), got {outer_rate}")
    if total_info_strands <= 0:
        raise ValueError(f"total_info_strands must be > 0, got {total_info_strands}")
    
    estimated_total_C = total_info_strands * (1 - outer_rate) / outer_rate
    estimated_total_encoded = total_info_strands + estimated_total_C
    num_packets = max(1, (int(estimated_total_encoded) + max_encoded_strands_per_packet - 1) // max_encoded_strands_per_packet)
    
    info_per_packet = _distribute_evenly(total_info_strands, num_packets)
    redundancy_per_packet = [rate_to_redundancy(outer_rate, M_i) for M_i in info_per_packet]
    
    while num_packets > 0:
        info_per_packet = _distribute_evenly(total_info_strands, num_packets)
        redundancy_per_packet = [rate_to_redundancy(outer_rate, M_i) for M_i in info_per_packet]
        
        max_packet_size = max((M_i + C_i for M_i, C_i in zip(info_per_packet, redundancy_per_packet)), default=0)
        if max_packet_size <= max_encoded_strands_per_packet:
            break
        num_packets += 1
    
    packet_plan = []
    total_redundancy = 0
    for packet_id, (info_count, redundancy_count) in enumerate(zip(info_per_packet, redundancy_per_packet)):
        encoded_count = info_count + redundancy_count
        if encoded_count > max_encoded_strands_per_packet:
            raise ValueError(
                f"Packet planning failed: packet {packet_id} needs {encoded_count} DNA sequences "
                f"(M={info_count}, C={redundancy_count}), limit is {max_encoded_strands_per_packet}."
            )
        packet_plan.append(
            {
                "packet_id": packet_id,
                "M": info_count,
                "outer_redundancy": redundancy_count,
                "N_strands": encoded_count,
            }
        )
        total_redundancy += redundancy_count
    
    return packet_plan, total_redundancy


def maximize_outer_redundancy_for_filtered_packets(
    packet_plan: List[dict],
    candidate_pool_size: int = FILTERED_CANDIDATE_POOL_SIZE,
) -> Tuple[List[dict], int]:
    """
    Expand each already-planned packet to the full filtered candidate pool.

    The packet count and information-sequence allocation are intentionally kept
    from the normal no-filtering plan, so toggling ``filtered`` does not change
    packetization. Each packet is instead extended to all 2^16 addressable DNA
    sequence indices:

        C_packet = candidate_pool_size - M_packet

    The user can then choose the number of sequences required by the requested
    outer rate from this full candidate pool.
    """
    if candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size must be > 0")

    maximized_plan = []
    total_redundancy = 0
    for packet in packet_plan:
        info_count = packet["M"]
        if info_count <= 0:
            raise ValueError("Each packet must have at least one information sequence.")
        if info_count > candidate_pool_size:
            raise ValueError(
                f"Packet {packet['packet_id']} has {info_count} information sequences, "
                f"which exceeds the filtered candidate pool of {candidate_pool_size}."
            )

        redundancy_count = candidate_pool_size - info_count
        maximized_packet = dict(packet)
        maximized_packet["selection_required"] = packet["N_strands"]
        maximized_packet["outer_redundancy"] = redundancy_count
        maximized_packet["N_strands"] = candidate_pool_size
        maximized_plan.append(maximized_packet)
        total_redundancy += redundancy_count

    return maximized_plan, total_redundancy


def split_file_into_packets(data_bits: np.ndarray, k: int, packet_info_strands: List[int] | None = None) -> List[Tuple[np.ndarray, int]]:
    """
    Split file data into packets, each with max 2^12 information sequences.
    
    Args:
        data_bits (np.ndarray): 1D array of binary bits from the file
        k (int): Information bits length per DNA sequence (fixed parameter)
    
    Returns:
        List[Tuple[np.ndarray, int]]: List of (packet_data, M_packet) tuples
            where packet_data is the binary data for that packet
            and M_packet is the number of information sequences for that packet
    """
    if packet_info_strands is None:
        max_bits_per_packet = MAX_SEQS_PER_PACKET * k
        num_packets = (len(data_bits) + max_bits_per_packet - 1) // max_bits_per_packet
        packet_info_strands = []
        remaining_bits = len(data_bits)
        for _ in range(num_packets):
            bits_here = min(max_bits_per_packet, remaining_bits)
            packet_info_strands.append((bits_here + k - 1) // k)
            remaining_bits -= bits_here

    packets = []
    start_idx = 0
    total_info_strands = 0
    for M_packet in packet_info_strands:
        if M_packet <= 0:
            raise ValueError("Each packet must have at least one information sequence.")
        bits_for_packet = min(M_packet * k, len(data_bits) - start_idx)
        end_idx = start_idx + max(bits_for_packet, 0)
        packet_bits = data_bits[start_idx:end_idx].copy()
        packets.append((packet_bits, M_packet))
        start_idx = end_idx
        total_info_strands += M_packet

    expected_info_strands = (len(data_bits) + k - 1) // k if len(data_bits) else 0
    if total_info_strands != expected_info_strands:
        raise ValueError(
            f"Packet split mismatch: planned {total_info_strands} information sequences, "
            f"but data requires {expected_info_strands}."
        )

    return packets


def group_strands_by_packet_id(
    uhat_list: List[np.ndarray],
    packet_id_bits: int = PACKET_ID_BITS
) -> Dict[int, List[np.ndarray]]:
    """
    Group decoded DNA sequences by packet ID bits.
    
    Args:
        uhat_list (List[np.ndarray]): List of decoded binary sequences with packet ID prefix
        packet_id_bits (int): Number of bits for packet ID
    
    Returns:
        Dict[int, List[np.ndarray]]: Dictionary mapping packet_id -> list of sequences for that packet
    """
    packets_dict: Dict[int, List[np.ndarray]] = {}
    
    for sequence in uhat_list:
        if len(sequence) < packet_id_bits:
            raise ValueError(
                f"DNA sequence too short: {len(sequence)} bits, "
                f"but packet_id_bits={packet_id_bits}"
            )
        
        packet_id = decode_packet_id_bits(sequence[:packet_id_bits])
        payload = sequence[packet_id_bits:].copy()
        
        packets_dict.setdefault(packet_id, []).append(payload)
    
    return packets_dict


def decode_packet_id_bits(packet_id_bits_array: np.ndarray) -> int:
    """Decode a packet ID stored as a binary array."""
    packet_id = 0
    for bit in packet_id_bits_array:
        packet_id = (packet_id << 1) | int(bit)
    return packet_id


def calculate_num_packets_required(file_size_bytes: int, k: int) -> int:
    """
    Calculate how many packets are needed to encode a file.
    
    Args:
        file_size_bytes (int): Size of file in bytes
        k (int): Information bits per DNA sequence
    
    Returns:
        int: Number of packets required
    """
    total_bits = file_size_bytes * 8
    bits_per_packet = MAX_SEQS_PER_PACKET * k
    return (total_bits + bits_per_packet - 1) // bits_per_packet


def levenshtein_distance(a: str, b: str, max_distance: int = None) -> int:
    """Compute Levenshtein distance between two equal-length strings.

    Uses an early stop when the distance exceeds max_distance.
    """
    if len(a) != len(b):
        raise ValueError("Levenshtein distance requires equal-length strings.")

    if max_distance is None:
        max_distance = len(a)

    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current_row = [i]
        min_current = i
        for j, cb in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (ca != cb)
            cost = min(insert_cost, delete_cost, replace_cost)
            current_row.append(cost)
            min_current = min(min_current, cost)

        if min_current > max_distance:
            return max_distance + 1
        previous_row = current_row

    return previous_row[-1]
