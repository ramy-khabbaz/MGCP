import numpy as np
import matplotlib.pyplot as plt
import time
import os
from multiprocessing import Pool, cpu_count
from mgcp.binary.encode import encode as encode_binary
from mgcp.binary.decode import decode as decode_binary
from mgcp.utils.binary_channel import binary_channel
from mgcp.utils.loader import load_codebook_binary


def _process_chunk(args):
    pe, chunk_size, Pd_Pi_Ps_ratio, message_length, l, parities_count, marker_period = args
    Pd, Pi, Ps = [r * pe for r in Pd_Pi_Ps_ratio]
    success, fail, error = 0, 0, 0
    decoding_time = 0.0
    rng = np.random.default_rng()

    for _ in range(chunk_size):
        u = rng.integers(0, 2, message_length).tolist()
        encoded, meta = encode_binary(u, l, parities_count, marker_period)
        noisy = binary_channel(encoded, Pd, Pi, Ps)

        start = time.perf_counter()
        decoded = decode_binary(noisy, meta)
        decoding_time += time.perf_counter() - start

        if decoded == u:
            success += 1
        elif not decoded:
            fail += 1
        else:
            error += 1

    return success, fail, error, decoding_time


def error_rate_vs_coderate(
    message_length,
    parities_count_list,
    l,
    marker_period,
    Pe=0.01,
    num_iterations=1000,
    Pd_Pi_Ps_ratio=(0.333, 0.333, 0.333),
    save_plot=True,
    processes=(cpu_count() // 2)
):

    cwd = os.getcwd()
    log_path = os.path.join(cwd, f"error_rate_vs_coderate_Pe{Pe}_m{marker_period}.txt")
    print(f"Logging results to: {log_path}")

    total_iter = int(num_iterations)
    CHUNK_SIZE = max(1, total_iter // (processes * 4))
    codebook = load_codebook_binary()
    results = []
    whole_start = time.perf_counter()

    with open(log_path, "w", encoding="utf-8") as log_file:
        header = (
            f"--- MGC+ Error Rate Vs. Code Rate Simulation ---\n"
            f"Message length: {message_length}\n"
            f"Block size l: {l}\n"
            f"Marker period: {marker_period}\n"
            f"Fixed Pe: {Pe}\n"
            f"Iterations per config: {num_iterations}\n"
            f"{'-'*50}\n"
        )
        print(header)
        log_file.write(header)

        for parities_count in parities_count_list:

            # --- Compute code rate ---
            if marker_period != 0:
                code_rate = message_length / (
                    message_length
                    + ((int(np.ceil(message_length / l)) + parities_count) / marker_period) * 3
                    + parities_count * l
                    + len(codebook[0])
                )
            else:
                code_rate = message_length / (
                    message_length + parities_count * l + len(codebook[0])
                )

            # --- Multiprocessing tasks ---
            num_chunks = (total_iter + CHUNK_SIZE - 1) // CHUNK_SIZE
            tasks = [
                (Pe, min(CHUNK_SIZE, total_iter - idx * CHUNK_SIZE),
                 Pd_Pi_Ps_ratio, message_length, l, parities_count, marker_period)
                for idx in range(num_chunks)
            ]

            try:
                with Pool(processes=processes) as pool:
                    results_chunks = pool.map(_process_chunk, tasks)
            except RuntimeError:
                print("[Warning] Multiprocessing unavailable. Running sequentially.")
                results_chunks = [_process_chunk(task) for task in tasks]

            total_S = sum(r[0] for r in results_chunks)
            total_F = sum(r[1] for r in results_chunks)
            total_E = sum(r[2] for r in results_chunks)
            total_T = sum(r[3] for r in results_chunks)
            total = total_S + total_F + total_E

            error_rate = 1 - total_S / total
            fail_rate = total_F / total
            pure_error_rate = total_E / total
            avg_time = total_T / total

            results.append({
                "CodeRate": code_rate,
                "TotalErrorRate": error_rate,
                "FailureRate": fail_rate,
                "ErrorRate": pure_error_rate,
                "AvgDecodeTime": avg_time,
                "parities_count": parities_count,
            })

            msg = (
                f"p={parities_count}, R={code_rate:.4f} | "
                f"Err={error_rate:.6f}, Fail={fail_rate:.6f}, "
                f"ErrOnly={pure_error_rate:.6f}, Time={avg_time:.4f}s"
            )
            print(msg)
            print(msg, file=log_file, flush=True)

        total_time_msg = f"\nTotal simulation time: {time.perf_counter() - whole_start:.2f} seconds\n"
        print(total_time_msg)
        print(total_time_msg, file=log_file, flush=True)

    # --- Plot ---
    rates = [r["CodeRate"] for r in results]
    errors = [r["TotalErrorRate"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(rates, errors, "o-", label=f"Error Rate (Pe={Pe})")
    plt.xlabel("Code Rate (R)")
    plt.ylabel("Error Rate")
    plt.title(
        f"MGC+ FER vs Code Rate (k={message_length}, l={l}, marker={marker_period})"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_plot:
        plot_path = os.path.join(cwd, f"mgcp_binary_FERvsCoderate_Pe{Pe}_m{marker_period}.png")
        plt.savefig(plot_path, dpi=1200)
        print(f"Plot saved to: {plot_path}")

    plt.show()
    return results
