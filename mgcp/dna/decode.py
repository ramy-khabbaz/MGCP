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

REALTIME_DRIFT_FLOOR = 0.01


def _resolve_explicit_decoder_error_rates(Pd=None, Pi=None, Ps=None):
    provided = [rate is not None for rate in (Pd, Pi, Ps)]
    if not any(provided):
        return None
    if not all(provided):
        raise ValueError("Provide Pd, Pi, and Ps together, or leave all three unset.")

    Pd, Pi, Ps = float(Pd), float(Pi), float(Ps)
    rates = {"Pd": Pd, "Pi": Pi, "Ps": Ps}
    for name, rate in rates.items():
        if rate < 0 or rate >= 1:
            raise ValueError(f"{name} must be in [0, 1), got {rate}.")

    if Pd + Pi + Ps >= 1:
        raise ValueError(f"Pd + Pi + Ps must be less than 1, got {Pd + Pi + Ps}.")

    return Pd, Pi, Ps


def _decode_once(corrupted_sequence, metadata, profile):
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
    Pd, Pi, Ps = profile
    P0 = 0.25

    match marker_period:
        case 0:
            len_last = (k - 1) % l + 1
            lim = 5
            lambda_depths = (1, 1, 0, 0, 0)

            Patterns = _get_patterns_cached(l, K, len_last, lim, parities_count, lambda_depths)

            decoded_message, _ = GCP_Decode_DNA_brute(
                corrupted_sequence, n, k, l, N, K, parities_count,
                q, len_last, lim, Patterns, codebook, dmin, opt=False
            )

        case 1:
            decoded_message, _, _, _ = MGCP_Decode_DNA_p1(
                corrupted_sequence, n, l, N, K, parities_count, c2, q, maxSize,
                P0, Pd, Pi, Ps, codebook, dmin
            )

        case 2:
            decoded_message, _, _, _, _ = MGCP_Decode_DNA_p2(
                corrupted_sequence, n, l, N, K, parities_count, c2, q, maxSize,
                2, P0, Pd, Pi, Ps, codebook, dmin
            )

        case _:
            raise ValueError(f"Unsupported marker_period: {marker_period}")

    return decoded_message


def _profile_from_length_drift_floor(received_length, expected_length):
    """Estimate one drift-only profile with a fixed 1% nonzero floor."""
    expected_length = max(int(expected_length), 1)
    net_drift = int(received_length) - expected_length
    drift_rate = abs(net_drift) / expected_length
    floor = REALTIME_DRIFT_FLOOR

    if net_drift > 0:
        deletion_rate = floor
        insertion_rate = floor + drift_rate
        substitution_rate = insertion_rate
    elif net_drift < 0:
        insertion_rate = floor
        deletion_rate = floor + drift_rate
        substitution_rate = deletion_rate
    else:
        deletion_rate = insertion_rate = substitution_rate = floor

    profile = (deletion_rate, insertion_rate, substitution_rate)
    total = sum(profile)
    if total >= 0.95:
        profile = tuple(rate * 0.95 / total for rate in profile)

    return profile, {
        "expected_length": expected_length,
        "received_length": int(received_length),
        "net_drift": net_drift,
        "drift_rate": drift_rate,
        "floor": floor,
        "target_Pe": sum(profile),
    }


def decode(
    corrupted_sequence,
    metadata=None,
    meta_path=None,
    Pd=None,
    Pi=None,
    Ps=None,
    return_diagnostics=False,
):
    """Decode an MGC+ DNA sequence."""
    if metadata is None:
        meta_path = meta_path or os.path.join(os.getcwd(), "mgcp_encode_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("Metadata not found.")
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    explicit_profile = _resolve_explicit_decoder_error_rates(Pd=Pd, Pi=Pi, Ps=Ps)
    if explicit_profile is None:
        received = "".join(corrupted_sequence)
        profile, drift_diagnostics = _profile_from_length_drift_floor(
            len(received), metadata["n"]
        )
        mode = "estimated"
    else:
        profile = explicit_profile
        drift_diagnostics = {}
        mode = "explicit"

    decoded_message = _decode_once(corrupted_sequence, metadata, profile)
    diagnostics = {
        "mode": mode,
        "selected_profile": {
            "Pd": profile[0],
            "Pi": profile[1],
            "Ps": profile[2],
        },
        "estimated_Pe": sum(profile),
        **drift_diagnostics,
    }

    if return_diagnostics:
        return decoded_message, diagnostics
    return decoded_message
