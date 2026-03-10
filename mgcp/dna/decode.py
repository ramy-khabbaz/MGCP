from .GCP_Decode_DNA import GCP_Decode_DNA_brute
from .MGCP_Decode_DNA_p1 import MGCP_Decode_DNA_p1
from .MGCP_Decode_DNA_p2 import MGCP_Decode_DNA_p2
from .preCompute_Patterns import preCompute_Patterns
from mgcp.utils.loader import load_codebook_dna

import os
import pickle
import hashlib
from functools import lru_cache
import json

# ============================================================
#               PATTERN CACHE SYSTEM
# ============================================================

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".mgcp_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


def _patterns_cache_key(l, K, len_last, lim, parities_count, lambda_depths):
    key = pickle.dumps((l, K, len_last, lim, parities_count, lambda_depths))
    return hashlib.sha256(key).hexdigest()[:16]


@lru_cache(maxsize=16)
def _get_patterns_cached(l, K, len_last, lim, parities_count, lambda_depths_tuple):
    """Memory + disk cached Patterns loader."""
    lambda_depths = list(lambda_depths_tuple)
    key = _patterns_cache_key(l, K, len_last, lim, parities_count, lambda_depths)
    cache_file = os.path.join(_CACHE_DIR, f"patterns_{key}.pkl")

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    Patterns = preCompute_Patterns(lambda_depths, K, len_last, lim, parities_count)

    with open(cache_file, "wb") as f:
        pickle.dump(Patterns, f)

    return Patterns


# ============================================================
#                       DECODER
# ============================================================

def decode(encoded_message, metadata=None, meta_path=None):
    if metadata is None:
        meta_path = meta_path or os.path.join(os.getcwd(), "mgcp_encode_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("Metadata not found.")
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    l = metadata["l"]
    parities_count = metadata["parities_count"]
    marker_period = metadata["marker_period"]
    n = metadata["n"]
    k = metadata["k"]
    N = metadata["N"]
    K = metadata["K"]
    q = metadata["q"]

    codebook = load_codebook_dna()
    dmin = 5
    c2 = 1
    maxSize = 1000
    Pe = 0.05
    Pd, Pi, Ps = 0.447 * Pe, 0.026 * Pe, 0.527 * Pe
    P0 = 0.25

    match marker_period:
        case 0:
            len_last = (k - 1) % l + 1
            lim = 5
            lambda_depths = (1, 1, 0, 0, 0)

            Patterns = _get_patterns_cached(l, K, len_last, lim, parities_count, lambda_depths)

            decoded, _ = GCP_Decode_DNA_brute(
                encoded_message, n, k, l, N, K, parities_count,
                q, len_last, lim, Patterns, codebook, dmin, opt=False
            )

        case 1:
            decoded, _, _, _ = MGCP_Decode_DNA_p1(
                encoded_message, n, l, N, K, parities_count, c2, q, maxSize,
                P0, Pd, Pi, Ps, codebook, dmin
            )

        case 2:
            decoded, _, _, _, _ = MGCP_Decode_DNA_p2(
                encoded_message, n, l, N, K, parities_count, c2, q, maxSize,
                2, P0, Pd, Pi, Ps, codebook, dmin
            )

        case _:
            raise ValueError(f"Unsupported marker_period: {marker_period}")

    return decoded
