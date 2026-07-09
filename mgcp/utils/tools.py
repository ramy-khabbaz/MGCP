import os
import numpy as np
import random
from multiprocessing import cpu_count
from numpy.random import default_rng
from mgcp.utils.DNA_iid_channel import DNA_iid_channel
from concurrent.futures import ProcessPoolExecutor


def generate_random_file(filename: str, size_bytes: int):
    rng = default_rng()
    data = rng.integers(0, 256, size_bytes, dtype=np.uint8)
    with open(filename, "wb") as f:
        f.write(data.tobytes())
    print(f"Random input file generated: {filename} ({size_bytes} bytes)")
    return filename


def _generate_chunk_reads(args):
    """
    Worker: generate all noisy reads for a chunk of (strand, n_reads) pairs.

    Batching multiple strands per task amortizes process-pool IPC overhead.
    Each strand gets an independent RNG seeded from its own seed value so
    results are reproducible regardless of chunk boundaries or worker count.
    """
    chunk, Pd, Pi, Ps = args
    reads = []
    errors = 0
    for strand, n_reads, seed in chunk:
        random.seed(seed)
        for _ in range(n_reads):
            noisy, n_err = DNA_iid_channel(list(strand), Pd, Pi, Ps)
            reads.append(noisy)
            errors += n_err
    return reads, errors


def error_generator(
    encoded_file,
    Pd, Pi, Ps,
    coverage,
    bias_sigma=0.25,
    output_path="reads.txt",
    n_workers=None,
    seed=None,
):
    """
    Simulate sequencing reads with global coverage constraint, bias, dropout,
    and iid base-level errors.

    Args:
        encoded_file (str): Path to file containing reference strands (one per line).
        Pd (float): Per-base deletion probability.
        Pi (float): Per-base insertion probability.
        Ps (float): Per-base substitution probability.
        coverage (float): Target mean coverage. Total reads = round(coverage * num_strands).
        bias_sigma (float): Lognormal sigma for coverage bias across strands.
        output_path (str): Where to write the simulated noisy reads.
        n_workers (int): Number of worker processes. Defaults to cpu_count().
        seed (int, optional): Reproducible coverage, channel, and shuffle seed.

    Returns:
        (output_path, stats_dict)
    """
    rng = np.random.default_rng(seed)

    # 1. Load strands
    with open(encoded_file, "r", encoding="utf-8") as f:
        strands = [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]
    num_strands = len(strands)

    # 2. Total read budget
    total_reads = int(round(coverage * num_strands))
    if total_reads == 0:
        with open(output_path, "w", encoding="utf-8") as f_out:
            pass
        return output_path, {
            "output_path": output_path,
            "per_strand_reads": np.zeros(num_strands, dtype=int),
            "dropout_rate": 1.0,
            "total_reads": 0,
            "mean_coverage": 0.0,
            "total_errors": 0,
        }

    # 3-5. Lognormal bias → per-strand read counts
    mu = -0.5 * (bias_sigma ** 2)
    weights = rng.lognormal(mean=mu, sigma=bias_sigma, size=num_strands)
    probs = weights / weights.sum()
    per_strand_reads = rng.multinomial(total_reads, probs)

    print(f"Simulating sequencing for {num_strands} strands (depth ≈ {coverage})...")

    # 6. Build per-strand tasks (skip strands with 0 reads)
    seeds = rng.integers(0, 2**31, size=num_strands)
    active = [
        (strands[i], int(per_strand_reads[i]), int(seeds[i]))
        for i in range(num_strands)
        if per_strand_reads[i] > 0
    ]

    n_workers = n_workers or cpu_count()

    # Chunk strands into n_workers batches so each worker gets a substantial
    # amount of work per submit. One task per strand (the old approach) means
    # thousands of tiny round-trips through the process pool, keeping IPC
    # overhead high and CPU utilization low. Chunking amortizes that cost.
    #
    # We target n_workers chunks (one per worker), but cap chunk size so a
    # single slow strand doesn't bottleneck everything — use 4x more chunks
    # than workers to give the pool something to steal if one chunk finishes
    # early. This balances load without excessive IPC.
    n_chunks = min(n_workers * 4, len(active))
    if n_chunks == 0:
        n_chunks = 1

    chunk_size = max(1, len(active) // n_chunks)
    chunks = [active[i:i + chunk_size] for i in range(0, len(active), chunk_size)]

    simulated_reads = []
    total_errors = 0

    if n_workers == 1 or len(chunks) == 1:
        # Single-process fallback: avoids pool overhead entirely for small jobs
        # or when called from inside an existing worker (nested pool on Linux is
        # legal with fork but wastes resources; skip it when not needed).
        for chunk in chunks:
            reads, errs = _generate_chunk_reads((chunk, Pd, Pi, Ps))
            simulated_reads.extend(reads)
            total_errors += errs
    else:
        tasks = [(chunk, Pd, Pi, Ps) for chunk in chunks]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for reads, errs in pool.map(_generate_chunk_reads, tasks):
                simulated_reads.extend(reads)
                total_errors += errs

    # 7. Shuffle to mimic sequencing read order randomness
    random.Random(seed).shuffle(simulated_reads)

    # 8. Write
    with open(output_path, "w", encoding="utf-8") as f_out:
        for read in simulated_reads:
            f_out.write(read + "\n")

    # 9. Diagnostics
    dropout = np.mean(per_strand_reads == 0)
    mean_cov_realized = per_strand_reads.mean()
    print(f"Dropout rate {100*dropout:.2f}%.")

    return output_path, {
        "output_path": output_path,
        "per_strand_reads": per_strand_reads,
        "dropout_rate": dropout,
        "total_reads": int(per_strand_reads.sum()),
        "mean_coverage": float(mean_cov_realized),
        "total_errors": total_errors,
    }


def compare_decoded_with_original(original_path, decoded_path):
    """
    Compare decoded file with the original input.

    Returns:
        (success: bool, reason: str)
    """
    if not decoded_path or not os.path.exists(decoded_path):
        return False, "Decoding failure — no decoded file produced."

    with open(original_path, "rb") as f1, open(decoded_path, "rb") as f2:
        original_data = f1.read()
        decoded_data = f2.read()
        if original_data != decoded_data:
            return False, "File mismatch — decoded file differs from original."

    return True, "Decoding success — files are identical"
