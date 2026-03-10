"""
MGC+ DNA Codec Full Pipeline Demo
=================================

This demo:
1. Generates a random binary input file
2. Encodes it using MGC+ DNA codec
3. Simulates sequencing reads with errors
   (insertions, deletions, substitutions, and lognormal coverage)
4. Decodes the noisy reads
5. Compares decoded file with original to verify success
"""

import mgcp
from multiprocessing import cpu_count
from mgcp.utils.tools import error_generator, generate_random_file, compare_decoded_with_original
from demo_utils import cluster_cdhit, kalign_and_consensus

# === Parameters ===
input_file_size = 5000 # in Bytes
max_length = 126
inner_redundancy = 4
outer_redundancy = 550
Pe = 0.05
Pd, Pi, Ps =  0.447 * Pe, 0.026 * Pe, 0.527 * Pe # example error rates
coverage = 10  # average number of reads per strand

# === MAIN PIPELINE ===
if __name__ == "__main__":
   # Step 1: Generate input file
   original_filename = "random_input.bin"
   input_file = generate_random_file(original_filename, input_file_size)

   # === Step 2: Encode ===
   mgcp.dna.codec.encode(original_filename, max_length, inner_redundancy, outer_redundancy, input_path=None, useMarker=True, filtered=False)
   encoded_file = "encoded_file.txt"

   # === Step 3: Simulate sequencing ===
   reads_file = error_generator(encoded_file, Pd, Pi, Ps, coverage, bias_sigma=0.25, output_path="reads.txt")[0]

   # [Optional] === Step 4: Clustering using CD-HIT ===
   clusters, stats = cluster_cdhit(reads_file, identity=0.85, word_size=6, threads=cpu_count(), memory=None)

   # [Optional] === Step 5: Alignment using Kalign + consensus ===
   kalign_and_consensus(clusters, n_workers=cpu_count())
   reads_file = "consensuses.txt"

   # === Step 4: Decode ===
   decoded_file = mgcp.dna.codec.decode(reads_file, processes=cpu_count())

   # === Step 5: Compare ===
   success, reason = compare_decoded_with_original(original_filename, "decoded_output.bin")
   print("\n=== SUMMARY ===")
   print(reason)