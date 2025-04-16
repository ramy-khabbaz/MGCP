import argparse
import numpy as np
import time
from multiprocessing import Pool, cpu_count
import math
from MGCP_Encode_DNA_p2 import MGCP_Encode_DNA_p2
from MGCP_Encode_DNA_p1 import MGCP_Encode_DNA_p1
from DNA_iid_channel import DNA_iid_channel
from MGCP_Decode_DNA_p2 import MGCP_Decode_DNA_p2
from MGCP_Decode_DNA_p1 import MGCP_Decode_DNA_p1
from CalculateProbas import CalculateProbas

def prob_value(value):
    """Converts input to float and checks it is between 0 and 1."""
    fvalue = float(value)
    if fvalue < 0 or fvalue > 1:
        raise argparse.ArgumentTypeError(f"{value} is not a valid probability. It must be between 0 and 1.")
    return fvalue

def marker_period_value(value):
    """Ensures marker period is either 1 or 2.""" 
    ivalue = int(value)
    if ivalue not in (1, 2):
        raise argparse.ArgumentTypeError("Marker period must be either 1 or 2.")
    return ivalue

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulation for MGCP Encoding/Decoding over a binary channel."
    )
    parser.add_argument("-i", "--iterations", type=int, required=True,
                        help="Number of iterations for the simulation")
    parser.add_argument("--Pd", type=prob_value, required=True,
                        help="Probability of deletion (between 0 and 1)")
    parser.add_argument("--Pi", type=prob_value, required=True,
                        help="Probability of insertion (between 0 and 1)")
    parser.add_argument("--Ps", type=prob_value, required=True,
                        help="Probability of substitution (between 0 and 1)")
    parser.add_argument("-k", type=int, required=True,
                        help="Parameter k (length of the message)")
    parser.add_argument("-l", type=int, required=True,
                        help="Parameter l (block length)")
    parser.add_argument("--c1_min", type=int, required=True,
                        help="Minimum value for c1 (range start)")
    parser.add_argument("--c1_max", type=int, required=True,
                        help="Maximum value for c1 (range end)")
    parser.add_argument("--c1_step", type=int, required=True,
                        help="Step for c1 values in the range")
    parser.add_argument("-c2", type=int, required=True,
                        help="Parameter c2")
    parser.add_argument("-t", type=int, required=True,
                        help="Parameter t (repeat count for check parity)")
    parser.add_argument("-m", "--marker_period", type=marker_period_value, required=True,
                        help="Marker period (only valid values: 1 or 2)")
    parser.add_argument("-o", "--output", type=str, default="results.txt",
                        help="Output file name to save the results")
    args = parser.parse_args()

    # Validate that the sum of probabilities does not exceed 1.
    if args.Pd + args.Pi + args.Ps > 1:
        parser.error("The sum of Pd, Pi, and Ps must not exceed 1.")

    # Generate the c1 values.
    c1_values = np.arange(args.c1_min, args.c1_max + 1, args.c1_step)

    if args.k % 2 != 0:
        parser.error("The length of the message should be even.")

    if args.l % 2 != 0:
        parser.error("The length of the block should be even.")

    if args.k % args.l != 0:
        parser.error("k must be divisible by l so that k/l is an integer.")
    
    # If marker period is 2, enforce that each c1 is even and k/l is an even integer.
    if args.marker_period == 2:
        if any(c1 % 2 != 0 for c1 in c1_values):
            parser.error("For marker period 2, all c1 values must be even.")
        if (args.k // args.l) % 2 != 0:
            parser.error("For marker period 2, k/l must be an even integer.")

    args.c1_values = c1_values
    return args

def initialize_globals(args):
    k = args.k
    l = args.l
    c2 = args.c2
    t = args.t
    marker_period = args.marker_period

    K = int(math.ceil(k / l))
    maxSize = 1000

    # Build lookup table L
    L = np.zeros((6, 5))
    for m_prime in range(6):
        for shift in range(-2, 3):
            L[m_prime, shift + 2] = CalculateProbas(m_prime, shift, (l // 2) * marker_period, 0.25, args.Pd, args.Pi, args.Ps)
    return k, l, c2, t, marker_period, K, maxSize, L

def process_chunk(args_tuple):
    # Unpack parameters
    c1, chunk_idx, chunk_size, total_iter, k, l, c2, t, marker_period, L, Pd, Pi, Ps, maxSize = args_tuple
    rng = np.random.default_rng()
    countS, countF, countE = 0, 0, 0
    decoding_time = 0
    
    for _ in range(chunk_size):
        u = rng.integers(0, 2, k).tolist()
        if marker_period == 1:
            x, n, N, K, q, U, X = MGCP_Encode_DNA_p1(u, l, c1, c2, t)
            y = DNA_iid_channel(x, Pd, Pi, Ps)
            start_time = time.perf_counter()
            uhat = MGCP_Decode_DNA_p1(y, n, k, l, N, K, c1, c2, q, t, L, maxSize)
            decoding_time += time.perf_counter() - start_time
        else:
            x, n, N, K, q, U, X = MGCP_Encode_DNA_p2(u, l, c1, c2, t)
            y = DNA_iid_channel(x, Pd, Pi, Ps)
            start_time = time.perf_counter()
            uhat = MGCP_Decode_DNA_p2(y, n, k, l, N, K, c1, c2, q, t, L, maxSize, marker_period)
            decoding_time += time.perf_counter() - start_time
        
        if uhat == u:
            countS += 1
        elif not uhat:
            countF += 1
        else:
            countE += 1

    return (c1, countS, countF, countE, decoding_time)

def main():
    args = parse_args()
    total_iter = args.iterations
    Pd, Pi, Ps = args.Pd, args.Pi, args.Ps
    k, l, c2, t, marker_period, K, maxSize, L = initialize_globals(args)
    
    c1_values = args.c1_values
    CHUNK_SIZE = max(1, 1000 // (cpu_count() * 4))

    with open(args.output, 'w') as log_file:
        for c1 in c1_values:
            tasks = []
            num_chunks = (total_iter + CHUNK_SIZE - 1) // CHUNK_SIZE
            for chunk_idx in range(num_chunks):
                start = chunk_idx * CHUNK_SIZE
                remaining = total_iter - start
                current_chunk = min(CHUNK_SIZE, remaining)
                tasks.append((c1, chunk_idx, current_chunk, total_iter, k, l, c2, t, marker_period, L, Pd, Pi, Ps, maxSize))
        
            whole_start = time.perf_counter()
            with Pool(processes=cpu_count()) as pool:
                results = pool.map(process_chunk, tasks)
            
            # Aggregate results per c1.
            results_dict = {c1: {'S': 0, 'F': 0, 'E': 0, 'time': 0} for c1 in c1_values}
            for result in results:
                c1, S, F, E, dt = result
                results_dict[c1]['S'] += S
                results_dict[c1]['F'] += F
                results_dict[c1]['E'] += E
                results_dict[c1]['time'] += dt

            total = results_dict[c1]['S'] + results_dict[c1]['F'] + results_dict[c1]['E']
            code_rate = k / (k + ((K+c1)/marker_period) * 4 + c1 * l + c2 * l * t)
            error_rate = 1 - results_dict[c1]['S'] / total
            avg_time = results_dict[c1]['time'] / total
            line = (f"c1: {c1}, Code Rate: {code_rate:.4f}, Error: {error_rate:.6f}, Avg Decoding Time: {avg_time:.6f}s")
            print(line)
            print(line, file=log_file, flush=True)
    
        total_execution = time.perf_counter() - whole_start
        summary_line = f"Total execution time: {total_execution:.2f} seconds"
        print(summary_line)
        print(summary_line, file=log_file, flush=True)

if __name__ == '__main__':
    main()