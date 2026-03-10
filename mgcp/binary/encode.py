from .GCP_Encode_binary import GCP_Encode_binary_brute
from .MGCP_Encode_Binary_p1 import MGCP_Encode_Binary_p1
from .MGCP_Encode_Binary_p2 import MGCP_Encode_Binary_p2
from mgcp.utils.loader import load_codebook_binary
import json
import os

_last_metadata = None

def encode(binary_message, l, parities_count, marker_period, export_json=False, export_path=None):
    """
    Encode a binary message using MGC+ encoding.
    
    Args:
        binary_message (list[int]): Binary input message.
        l (int): Block size.
        parities_count (int): Number of parity blocks.
        marker_period (int): Marker period (0, 1, or 2).
        export_json (bool, optional): If True, saves metadata to a JSON file.
        export_path (str, optional): Path for exporting metadata JSON (default: current directory).
    
    Returns:
        tuple: (encoded_message, metadata)
    """

    global _last_metadata

    k = len(binary_message)
    codebook = load_codebook_binary()

    # l must be even and must not exceed 8
    if l > 8:
        raise ValueError(f"Invalid block size l={l}. It must not exceed 8.")

    # marker_period must be 0, 1, or 2
    if marker_period not in (0, 1, 2):
        raise ValueError(f"Invalid marker_period={marker_period}. Must be 0, 1, or 2.")

    # k must be divisible by l
    if k % l != 0:
        raise ValueError(f"Length of binary message ({k}) must be divisible by l ({l}).")

    num_blocks = k // l

    # --- Special condition for marker_period = 2 ---
    if marker_period == 2:
        total = num_blocks + parities_count
        if total % 2 != 0:
            raise ValueError(
                f"For marker_period=2, total number of l-blocks ({num_blocks}) + "
                f"parities_count ({parities_count}) must be even (got {total})."
            )

    # --- Switch-case behavior depending on marker_period ---
    match marker_period:
        case 0:
            encoded_message, n, N, K, q, U, X = GCP_Encode_binary_brute(binary_message, l, parities_count, codebook)

        case 1:
            encoded_message, n, N, K, q, U, X = MGCP_Encode_Binary_p1(binary_message, l, parities_count, codebook)

        case 2:
            encoded_message, n, N, K, q, U, X = MGCP_Encode_Binary_p2(binary_message, l, parities_count, codebook)

    # --- After encoding ---
    metadata = {
        "l": l,
        "parities_count": parities_count,
        "marker_period": marker_period,
        "k": k,
        "K": K,
        "n": n,
        "N": N,
        "q": q
    }

    # --- Optional JSON export ---
    if export_json:
        meta_path = export_path or os.path.join(os.getcwd(), "mgcp_encode_meta.json")
        if metadata != _last_metadata:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            _last_metadata = metadata

    # Return both encoded message and metadata in memory
    return encoded_message, metadata
