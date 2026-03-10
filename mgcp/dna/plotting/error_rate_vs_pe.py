# mgcp/dna/plotting/run_error_rate_vs_pe.py
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math
from multiprocessing import Pool, cpu_count
from mgcp.dna.encode import encode as encode_dna
from mgcp.dna.decode import decode as decode_dna
from mgcp.utils.DNA_iid_channel import DNA_iid_channel
from mgcp.utils.loader import load_codebook_dna

def _process_chunk(args):
    """
    Worker function for multiprocessing.
    """
    pe, chunk_size, Pd_Pi_Ps_ratio, message_length, l, parities_count, marker_period = args
    Pd, Pi, Ps = [r * pe for r in Pd_Pi_Ps_ratio]
    success, fail, error = 0, 0, 0
    decoding_time = 0.0
    rng = np.random.default_rng()

    for _ in range(chunk_size):
        u = rng.integers(0, 2, message_length).tolist()
        encoded, meta = encode_dna(u, l, parities_count, marker_period)
        noisy = DNA_iid_channel(encoded, Pd, Pi, Ps)
        start = time.perf_counter()
        decoded = decode_dna(noisy, meta)
        decoding_time += time.perf_counter() - start

        if decoded == u:
            success += 1
        elif not decoded:
            fail += 1
        else:
            error += 1

    return pe, success, fail, error, decoding_time

def error_rate_vs_pe(
    message_length,
    l,
    parities_count,
    marker_period,
    Pe_range=np.arange(0.001, 0.011, 0.001),
    num_iterations=1000,
    Pd_Pi_Ps_ratio=(0.447, 0.026, 0.527),
    save_plot=True,
    processes = (cpu_count()//2)
):
    """
    Function to compute and plot MGC+ DNA error rate vs Pe.
    """
    if marker_period not in (0, 1, 2):
        raise ValueError("marker_period must be 0, 1, or 2.")

    cwd = os.getcwd()
    log_path = os.path.join(cwd, "error_rate_vs_pe_DNA_log.txt")
    print(f"Logging results to: {log_path}")

    total_iter = int(num_iterations)
    CHUNK_SIZE = max(1, total_iter // (processes * 4))

    # Load codebook (for code rate calculation)
    codebook = load_codebook_dna()
    if marker_period != 0:
        code_rate = message_length / (
            message_length
            + ((int(math.ceil(message_length / l)) + parities_count) / marker_period) * 4
            + parities_count * l
            + len(codebook[0]) * 2
        )
    else:
        code_rate = message_length / (
            message_length + parities_count * l + len(codebook[0]) * 2
        )

    results = []
    whole_start = time.perf_counter()

    # --- Log header ---
    with open(log_path, "w", encoding="utf-8") as log_file:
        header = (
            f"--- MGC+ DNA Error Rate Vs. Pe Simulation ---\n"
            f"Message length: {message_length}\n"
            f"Block size l: {l}\n"
            f"Parities count: {parities_count}\n"
            f"Marker period: {marker_period}\n"
            f"Code rate R: {code_rate:.4f}\n"
            f"Iterations per Pe: {num_iterations}\n"
            f"{'-'*40}\n"
        )
        print(header)
        log_file.write(header)

        for pe in Pe_range:
            # Prepare chunks
            num_chunks = (total_iter + CHUNK_SIZE - 1) // CHUNK_SIZE
            tasks = [
                (pe, min(CHUNK_SIZE, total_iter - idx * CHUNK_SIZE),
                 Pd_Pi_Ps_ratio, message_length, l, parities_count, marker_period)
                for idx in range(num_chunks)
            ]

            try:
                with Pool(processes=processes) as pool:
                    results_pe = pool.map(_process_chunk, tasks)
            except RuntimeError as e:
                print(f"[Warning] Multiprocessing unavailable. Running sequentially. Include if (__name__ == __main__) before running the script.")
                results_pe = [_process_chunk(task) for task in tasks]

            # Aggregate results
            total_S = sum(r[1] for r in results_pe)
            total_F = sum(r[2] for r in results_pe)
            total_E = sum(r[3] for r in results_pe)
            total_T = sum(r[4] for r in results_pe)
            total = total_S + total_F + total_E

            error_rate = 1 - total_S / total
            fail_rate = total_F / total
            pure_error_rate = total_E / total
            avg_time = total_T / total

            results.append({
                "Pe": pe,
                "TotalErrorRate": error_rate,
                "FailureRate": fail_rate,
                "ErrorRate": pure_error_rate,
                "AvgDecodeTime": avg_time,
            })

            msg = (
                f"Pe={pe:.3f} | TotalErr={error_rate:.6f} | "
                f"Fail={fail_rate:.6f} | ErrOnly={pure_error_rate:.6f} | "
                f"AvgTime={avg_time:.4f}s"
            )
            print(msg)
            print(msg, file=log_file, flush=True)

        total_time_msg = f"\nTotal simulation time: {time.perf_counter() - whole_start:.2f} seconds\n"
        print(total_time_msg)
        print(total_time_msg, file=log_file, flush=True)

    # --- Plot ---
    Pe_values = [r["Pe"] for r in results]
    ErrorRates = [r["TotalErrorRate"] for r in results]
    plt.figure(figsize=(8, 5))
    plt.plot(Pe_values, ErrorRates, "o-", label="Error Rate")
    plt.xlabel("Pe (Error Probability)")
    plt.ylabel("Error Rate")
    plt.title(
        f"MGC+ DNA FER vs Pe (k={message_length}, l={l}, p={parities_count}, m={marker_period}, R={code_rate:.3f})"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_plot:
        plot_path = os.path.join(cwd, f"mgcp_dna_FERvsPe_m{marker_period}.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Plot saved to: {plot_path}")

    plt.show()
    return results