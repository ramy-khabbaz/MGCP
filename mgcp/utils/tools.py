import os
import numpy as np
import random
from multiprocessing import cpu_count
from numpy.random import default_rng
from scipy.stats import lognorm
from mgcp.utils.DNA_iid_channel import DNA_iid_channel

def generate_random_file(filename: str, size_bytes: int):
    rng = default_rng()
    data = rng.integers(0, 256, size_bytes, dtype=np.uint8)
    with open(filename, "wb") as f:
        f.write(data.tobytes())
    print(f"Random input file generated: {filename} ({size_bytes} bytes)")
    return filename

def error_generator(
    encoded_file,
    Pd,
    Pi,
    Ps,
    coverage,
    bias_sigma=0.25,
    output_path="reads.txt"
):
    """
    Simulate sequencing reads with global coverage constraint, bias, dropout,
    and iid base-level errors.

    Args:
        encoded_file (str):
            Path to file containing reference strands.
            One strand (A/C/G/T string) per line.
        Pd (float):
            Per-base deletion probability.
        Pi (float):
            Per-base insertion probability.
        Ps (float):
            Per-base substitution probability.
        coverage (float):
            Target mean coverage n_bar. We will generate a total of
            round(coverage * num_strands) reads.
        bias_sigma (float):
            Lognormal sigma controlling bias in sampling probabilities.
            Larger means more uneven coverage.
        output_path (str):
            Where to write the simulated noisy reads.
    Returns:
        result (dict) with:
            "output_path": path to the file with all noisy reads
            "per_strand_reads": np.array of length num_strands
            "dropout_rate": fraction of strands with 0 reads
            "total_reads": total reads actually generated
            "mean_coverage": average reads per strand actually generated
    """

    rng = np.random.default_rng()
    
    # 1. Load strands
    with open(encoded_file, "r", encoding="utf-8") as f:
        strands = [line.strip() for line in f if line.strip() != ""]
    num_strands = len(strands)

    # 2. Decide total read budget
    total_reads = int(round(coverage * num_strands))

    # Edge case: coverage == 0
    if total_reads == 0:
        # No reads at all
        with open(output_path, "w", encoding="utf-8") as f_out:
            pass  # write empty file
        return {
            "output_path": output_path,
            "per_strand_reads": np.zeros(num_strands, dtype=int),
            "dropout_rate": 1.0,  # everyone dropped out
            "total_reads": 0,
            "mean_coverage": 0.0,
        }

    # 3. Draw lognormal weights to model bias
    mu = -0.5 * (bias_sigma ** 2)
    weights = rng.lognormal(mean=mu, sigma=bias_sigma, size=num_strands)

    # 4. Convert to sampling probabilities
    probs = weights / weights.sum()

    # 5. Multinomial draw for per-strand coverage
    per_strand_reads = rng.multinomial(total_reads, probs)

    print(f"Simulating sequencing for {num_strands} strands (sequencing depth ≈ {coverage})...")
    
    # 6. Generate noisy reads for each strand
    simulated_reads = []
    for strand_idx, strand in enumerate(strands):
        n_reads_i = per_strand_reads[strand_idx]
        for _ in range(n_reads_i):
            noisy = DNA_iid_channel(list(strand), Pd, Pi, Ps)
            simulated_reads.append(noisy)

    # 7. Shuffle to mimic sequencing read order randomness
    random.shuffle(simulated_reads)

    # 8. Write to output file
    with open(output_path, "w", encoding="utf-8") as f_out:
        for read in simulated_reads:
            f_out.write(read + "\n")

    # 9. Diagnostics
    dropout = np.mean(per_strand_reads == 0)
    mean_cov_realized = per_strand_reads.mean()

    # print(f"Simulated {num_strands} strands.")
    # print(f"Target mean coverage {coverage}, "
    #       f"total_reads budget {total_reads}.")
    # print(f"Realized mean coverage {mean_cov_realized:.3f}.")
    print(f"Dropout rate {100*dropout:.2f}%.")

    return output_path, {
        "output_path": output_path,
        "per_strand_reads": per_strand_reads,
        "dropout_rate": dropout,
        "total_reads": int(per_strand_reads.sum()),
        "mean_coverage": float(mean_cov_realized),
    }

def compare_decoded_with_original(original_path, decoded_path):
    """
    Compare decoded file with the original input.

    Returns:
        (success: bool, reason: str)
    """
    # Case 1: Decoding failed — no file produced
    if not decoded_path or not os.path.exists(decoded_path):
        return False, "Decoding failure — no decoded file produced."

    # Case 2: File exists but differs
    with open(original_path, "rb") as f1, open(decoded_path, "rb") as f2:
        original_data = f1.read()
        decoded_data = f2.read()
        if original_data != decoded_data:
            return False, "File mismatch — decoded file differs from original."

    # Case 3: Success
    return True, "Decoding success — files are identical"