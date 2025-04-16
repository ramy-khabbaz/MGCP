import argparse
import numpy as np
import time
from multiprocessing import Pool, cpu_count
import math
from MGCP_Encode_p2 import MGCP_Encode_p2
from MGCP_Encode_p1 import MGCP_Encode_p1
from binary_channel import binary_channel
from MGCP_Decode_p2 import MGCP_Decode_p2
from MGCP_Decode_p1 import MGCP_Decode_p1
from CalculateProbas import CalculateProbas

def float_value(value):
    try:
        return float(value)
    except:
        raise argparse.ArgumentTypeError("Value must be a float.")

def marker_period_value(value):
    ivalue = int(value)
    if ivalue not in (1, 2):
        raise argparse.ArgumentTypeError("Marker period must be either 1 or 2.")
    return ivalue

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulation for MGCP Encoding/Decoding over a binary channel with Pe range and separate coefficients for Pd, Pi, and Ps."
    )
    parser.add_argument("-i", "--iterations", type=int, required=True,
                        help="Number of iterations for the simulation")
    parser.add_argument("--Pe_min", type=float_value, required=True,
                        help="Minimum value for Pe (error probability)")
    parser.add_argument("--Pe_max", type=float_value, required=True,
                        help="Maximum value for Pe (error probability)")
    parser.add_argument("--Pe_step", type=float_value, required=True,
                        help="Step value for Pe")
    parser.add_argument("--coeffPd", type=float_value, required=True,
                        help="Coefficient for Pd (so that Pd = coeffPd * Pe)")
    parser.add_argument("--coeffPi", type=float_value, required=True,
                        help="Coefficient for Pi (so that Pi = coeffPi * Pe)")
    parser.add_argument("--coeffPs", type=float_value, required=True,
                        help="Coefficient for Ps (so that Ps = coeffPs * Pe)")
    parser.add_argument("-k", type=int, required=True,
                        help="Parameter k (length of the message)")
    parser.add_argument("-c1", type=int, required=True,
                        help="Parameter c1")
    parser.add_argument("-l", type=int, required=True,
                        help="Parameter l (block length)")
    parser.add_argument("-c2", type=int, required=True,
                        help="Parameter c2")
    parser.add_argument("-t", type=int, required=True,
                        help="Parameter t (repeat count for check parity)")
    parser.add_argument("-m", "--marker_period", type=marker_period_value, required=True,
                        help="Marker period (only valid values: 1 or 2)")
    parser.add_argument("-o", "--output", type=str, default="results.txt",
                        help="Output file name for saving the results")
    args = parser.parse_args()

    # Generate the Pe values range.
    Pe_values = np.arange(args.Pe_min, args.Pe_max+args.Pe_step , args.Pe_step)
    
    # Check that for every Pe the sum of probabilities is ≤ 1.
    total_coeff = args.coeffPd + args.coeffPi + args.coeffPs
    for pe in Pe_values:
        if total_coeff * pe > 1:
            parser.error(
                f"For Pe={pe:.4f}, (coeffPd+coeffPi+coeffPs)*Pe = {total_coeff*pe:.4f} exceeds 1. "
                "Please adjust the coefficients or the Pe range."
            )
    
    args.Pe_values = Pe_values

    if args.k % args.l != 0:
        parser.error("k must be divisible by l so that k/l is an integer.")

    # If marker period is 2, enforce that each c1 is even and k/l is an even integer.
    if args.marker_period == 2:
        if args.c1 % 2 != 0:
            parser.error("For marker period 2, all c1 values must be even.")
        if (args.k // args.l) % 2 != 0:
            parser.error("For marker period 2, k/l must be an even integer.")

    return args

def initialize_globals(args):

    k = args.k
    c1 = args.c1
    c2 = args.c2
    l = args.l
    t = args.t
    marker_period = args.marker_period
    K = int(math.ceil(k / l))
    maxSize = 1000
    P0 = 0.5
    return k, c1, c2, l, t, marker_period, K, maxSize, P0

def process_chunk(args_tuple):

    (pe, chunk_idx, chunk_size, total_iter, k, c1, c2, l, t, marker_period, P0, coeffPd, coeffPi, coeffPs, maxSize) = args_tuple
    rng = np.random.default_rng()
    countS, countF, countE = 0, 0, 0
    decoding_time = 0

    # Calculate error probabilities for this Pe.
    Pd = coeffPd * pe
    Pi = coeffPi * pe
    Ps = coeffPs * pe

    # Build lookup table L for these probabilities.
    L = np.zeros((8, 5))
    for m_prime in range(8):
        for shift in range(-2, 3):
            L[m_prime, shift + 2] = CalculateProbas(m_prime, shift, l * marker_period, P0, Pd, Pi, Ps)

    for _ in range(chunk_size):
        u = rng.integers(0, 2, k).tolist()
        if marker_period == 1:
            x, n, N, K, q, U, X = MGCP_Encode_p1(u, l, c1, c2, t)
            y = binary_channel(x, Pd, Pi, Ps)
            start_time = time.perf_counter()
            uhat = MGCP_Decode_p1(y, n, k, l, N, K, c1, c2, q, t, L, maxSize)
            decoding_time += time.perf_counter() - start_time
        else:
            x, n, N, K, q, U, X = MGCP_Encode_p2(u, l, c1, c2, t)
            y = binary_channel(x, Pd, Pi, Ps)
            start_time = time.perf_counter()
            uhat = MGCP_Decode_p2(y, n, k, l, N, K, c1, c2, q, t, L, maxSize, marker_period)
            decoding_time += time.perf_counter() - start_time

        if uhat == u:
            countS += 1
        elif not uhat:
            countF += 1
        else:
            countE += 1

    return (pe, countS, countF, countE, decoding_time)

def main():
    args = parse_args()
    total_iter = args.iterations
    Pe_values = args.Pe_values
    coeffPd = args.coeffPd
    coeffPi = args.coeffPi
    coeffPs = args.coeffPs

    # Fixed globals.
    k, c1, c2, l, t, marker_period, K, maxSize, P0 = initialize_globals(args)
    
    # Precalculate a fixed code rate for display.
    fixed_code_rate = k / (k + ((K + c1) / marker_period) * 3 + c1 * l + c2 * l * t)
    
    # Set chunk size.
    CHUNK_SIZE = max(1, 1000 // (cpu_count() * 4))

    whole_start = time.perf_counter()

    with open(args.output, 'w') as log_file:
        # Loop over each Pe value.
        for pe in Pe_values:
            # Build a fresh tasks list for the current pe.
            tasks = []
            num_chunks = (total_iter + CHUNK_SIZE - 1) // CHUNK_SIZE
            for chunk_idx in range(num_chunks):
                remaining = total_iter - (chunk_idx * CHUNK_SIZE)
                current_chunk = min(CHUNK_SIZE, remaining)
                tasks.append((pe, chunk_idx, current_chunk, total_iter, k, c1, c2, l, t, marker_period, P0, coeffPd, coeffPi, coeffPs, maxSize))
            
            with Pool(processes=cpu_count()) as pool:
                results = pool.map(process_chunk, tasks)
            
            # Aggregate results
            results_dict = {pe: {'S':0, 'F':0, 'E':0, 'time':0} for pe in Pe_values}
            for result in results:
                pe, S, F, E, dt = result
                results_dict[pe]['S'] += S
                results_dict[pe]['F'] += F
                results_dict[pe]['E'] += E
                results_dict[pe]['time'] += dt
            
            # Calculate final metrics
            total = results_dict[pe]['S'] + results_dict[pe]['F'] + results_dict[pe]['E']
            error_rate = 1 - results_dict[pe]['S']/total
            avg_time = results_dict[pe]['time']/total
            
            # Display results
            msg=(f"Pe: {pe:.4f}, Error: {error_rate:.6f}, Time: {avg_time:.7f}s")
            print(msg)
            print(msg, file=log_file, flush=True)
        
        total_execution = time.perf_counter() - whole_start
        summary_line = (f"\nFixed Code Rate: {fixed_code_rate:.4f}, Total execution time: {total_execution:.2f} seconds")
        print(summary_line)
        print(summary_line, file=log_file, flush=True)

if __name__ == '__main__':
    main()
