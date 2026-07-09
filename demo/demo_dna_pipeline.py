"""
MGC+ DNA Codec Full Pipeline Demo
=================================

This demo:
1. Generates a random binary input file
2. Encodes it using MGC+ DNA codec
3. Channel simulator (noisy reads with insertions, deletions, substitutions, and lognormal coverage distribution)
4. Decodes the noisy reads
5. Compares decoded file with original to verify success

All generated files are grouped inside OUTPUT_DIR.
"""

import os
import sys
from multiprocessing import cpu_count
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mgcp
from mgcp.utils.tools import error_generator, generate_random_file, compare_decoded_with_original
from demo_utils import cluster_cdhit, kalign_and_consensus


# === Parameters ===
OUTPUT_DIR = "output"

input_file_size = 15000  # in Bytes
max_length = 152
inner_redundancy = 4
outer_rate = 0.8
Pe = 0.05 # total channel error rate
Pd, Pi, Ps = 0.447 * Pe, 0.026 * Pe, 0.527 * Pe  # del, ins, subs rates
coverage = 10  # average number of reads per DNA sequence
processes = cpu_count()


# === MAIN PIPELINE ===
if __name__ == "__main__":
   original_cwd = Path.cwd()
   output_dir = original_cwd / OUTPUT_DIR
   output_dir.mkdir(parents=True, exist_ok=True)

   os.chdir(output_dir)
   try:
      # Step 1: Generate input file
      original_filename = "random_input.bin"
      input_file = generate_random_file(original_filename, input_file_size)

      # === Step 2: Encode ===
      mgcp.dna.codec.encode(
         original_filename,
         max_length,
         inner_redundancy,
         outer_rate,
         input_path=None,
         useMarker=True,
         filtered=False,
         processes=processes,
      )
      encoded_file = "encoded_file.txt"

      # === Step 3: Channel simulator ===
      reads_file = error_generator(
         encoded_file,
         Pd,
         Pi,
         Ps,
         coverage,
         bias_sigma=0.25,
         output_path="reads.txt",
      )[0]

      # [Optional] === Step 4: Clustering using CD-HIT ===
      clusters, stats = cluster_cdhit(
         reads_file,
         identity=0.80,
         word_size=5,
         threads=processes,
         memory=None,
      )

      # [Optional] === Step 5: Alignment using Kalign + consensus ===
      kalign_and_consensus(clusters, n_workers=processes)
      reads_file = "consensuses.txt"

      # === Step 6: Decode ===
      decoded_file = mgcp.dna.codec.decode(
         reads_file,
         processes=processes,
      )

      # === Step 7: Compare ===
      success, reason = compare_decoded_with_original(original_filename, "decoded_output.bin")
      print("\n=== SUMMARY ===")
      print(reason)
      print(f"Output folder: {output_dir}")
   finally:
      os.chdir(original_cwd)
