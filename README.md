# MGCP — MGC+ coding for DNA and binary channels with insertion, deletion, and substitution (IDS) errors

MGCP is a Python package implementing the Marker Guess & Checl Plus (MGC+) family of encoders and decoders for both binary and DNA sequences. It contains:

- Encoders/decoders for binary and DNA sequences (`mgcp.binary`, `mgcp.dna`).
- File-level codec that encodes a binary file into a collection of DNA sequences and decodes it back from noisy DNA reads (`mgcp.dna.codec`).
- Utility modules for simulation, error models, and plotting (`mgcp.utils`).
- Command-line interface (`mgcp/cli`) that exposes the main workflows.
- Demos under `demo/` that show end-to-end examples (these require optional external tools).

This README documents installation, usage, CLI commands, demos and publishing guidance.

## Table of contents

- Features
- Installation (Python + optional system deps)
- Quickstart (import & CLI examples)
- Detailed module overview
- Demo & external tools
- Citing this work
- License

## Features

- Encode/decode at the bit level and the DNA level using MGC+ codes.
- Plotting helpers to benchmark frame error rate (FER) vs code 	rate or channel error rate (both DNA and binary).
- A single CLI entrypoint (`mgcp`) with subcommands for DNA, binary, and file codec flows.
- Demo scripts showing how to run full pipelines, clustering and consensus building.

## Installation

Prerequisites:

- Python 3.9 or newer.
- A working C/Python toolchain only if you need to build some optional native deps.

clone and install locally:

```bash
git clone https://github.com/ramy-khabbaz/mgcp.git
cd mgcp
pip install -e .
```

### Optional demo/system dependencies

The package declares all runtime dependencies in `setup.cfg`. Some demo scripts require external, non-Python tools (listed below). If you only want to use the core library/CLI you can install as above. To enable demo-only features (clustering, MSA), install the demo extras:

```bash
pip install -e '.[demo]'
```

What the demo extras install:

- `kalign` — Python wrapper for Kalign MSA
- `pycdhit` — lightweight wrapper for CD-HIT
- `psutil`, `tqdm`

Note: `kalign` and `pycdhit` requires system packages or binaries (see Demo & external tools below).

## Quickstart

### Importing the library

```py
import mgcp
print(mgcp.__version__)

# programmatic usage example (file codec)
from mgcp.dna.codec import encode as codec_encode
codec_encode(file_name='data.bin', max_length=120, inner_redundancy=4, outer_redundancy=200)
```

### Command-line interface

MGCP exposes a single CLI entrypoint `mgcp` that groups subcommands.

Top-level help:

```bash
mgcp --help
```

Subcommands

- DNA-level: `mgcp dna ...`
- Binary-level: `mgcp binary ...` (encode/decode and plotting)
- File-level codec: `mgcp codec ...` (encode/decode files to/from DNA sequences)

Examples

Output a detailed list of the MGC+ encoding parameters for different modules:

```bash
mgcp binary encode --help
mgcp dna encode --help
mgcp codec encode --help
```

Encode a single binary message into a binary codeword, with the block size (symbol length) set to 4 bits, 4 guess parities added, and the marker period set to 2:

```bash
mgcp binary encode "0101010011110110" 4 4 2
```

Recover the binary message from a corrupted sequence (the 6th and 7th bits of the codeword are deleted and the 20th is substituted):

```bash
mgcp binary decode "0101000011111011010111110111001010101100010010100101101100100011"
```

Encode a single binary message into a single DNA sequence, with the block size (symbol length) set to 4 bits, 4 guess parities added, and the marker period set to 0 (no markers):

```bash
mgcp dna encode "0101010011110110" 4 4 0
```

Recover the binary message from a corrupted DNA sequence (substitutions: 2nd (T->A) and 17th (C->T) bases, deletions: 10th and 11th bases, insertion: 'G' is inserted at the 4th position):

```bash
mgcp dna decode "TATGAGGTCGGTTTCTCTGATTGTGTT"
```

Encode a binary file into a collection of DNA sequences (file-level DNA-MGC+ codec). Here, the file is encoded with a target oligo length of 120, the inner code has 4 guess parities and doesn't include markers, and the outer code adds 200 redundant sequences:

```bash
mgcp codec encode "data.bin" 120 4 200 --input-path ./ --no-marker
```

Recover the binary file from noisy DNA reads (reads.txt) using 4 CPU cores for parallel decoding:

```bash
mgcp codec decode "reads.txt" --input-path ./ --processes 4
```

### Plotting

Both `mgcp dna` and `mgcp binary` include `plot` subcommands to generate FER vs code rate or FER vs channel error rate. Example:

```bash
mgcp dna plot fer-vs-coderate 1000 126 1 "1,2,3,4" --pe 0.01 --num-iterations 500
```

## Detailed module overview

- `mgcp.binary` — binary-level encoding/decoding primitives and utilities.
- `mgcp.dna` — binary input to DNA sequence encoding, decoding, and helper pipelines.
- `mgcp.dna.codec` — high-level file codec (binary file -> DNA sequence and Noisy reads -> binary file).
- `mgcp.utils` — helper modules: `tools.py` (random file generation, error models), `loader.py`, `binary_channel.py`, and plotting utilities.
- `mgcp.cli` — `main.py` registers Typer application and subcommands implemented in `dna_cli.py`, `binary_cli.py`, `codec_cli.py`.

For programmatic use, import the submodule you need and call the functions directly. Examples can be found in `demo/`.

## Demo & external tools

The `demo/` folder demonstrates the full pipeline: encode a file, simulate sequencing errors, cluster reads (CD-HIT), align clusters (Kalign), generate consensus sequences, and decode back.

### External tools used by demos

- CD-HIT (cd-hit-est) — clustering. Install the executable and ensure `cd-hit-est` is on PATH. The demo extras install only Python wrappers; the native binary must be installed separately.
- Kalign — multiple sequence aligner. Install the Kalign binary (or Kalign3) and ensure it is on PATH. The Python wrapper may still require the Kalign executable.

References and links

- CD-HIT — Li, W. & Godzik, A. (2006). Cd-hit: a fast program for clustering and comparing large sets of protein or nucleotide sequences. Bioinformatics 22(13):1658–1659. DOI: https://doi.org/10.1093/bioinformatics/btl158. Project: https://github.com/weizhongli/cdhit
- Kalign — Lassmann, T. & Sonnhammer, E. L. L. (2005). Kalign—an accurate and fast multiple sequence alignment algorithm. BMC Bioinformatics 6:298. DOI: https://doi.org/10.1186/1471-2105-6-298. Kalign homepage: https://msa.sbc.su.se/kalign/ · Kalign3: https://github.com/TimoLassmann/kalign3

Note: check each tool's README for platform-specific dependencies and recommended installation methods.

### Example demo outline

1. `mgcp.dna.codec.encode` to generate `encoded_file.txt` (oligos list).
2. `mgcp.utils.tools.error_generator` to generate reads with IDS errors.
3. Run CD-HIT on the reads to cluster similar reads together (`cd-hit-est` or the `pycdhit` helper).
4. For each cluster, run Kalign to apply multiple sequence alignment to the reads and produce a consensus sequence.
5. Feed consensus sequences into `mgcp.dna.codec.decode` to recover the original file.

The demo scripts in `demo/` show concrete invocations. To run demos, install the demo extras and ensure `cd-hit-est` and `kalign` are installed on your system.


## Citing this work

If you use MGCP in your research, please cite the project. A machine-readable citation file is provided in `CITATION.cff`.

BibTeX example:

```bibtex
@software{mgcp2025,
	title = {MGC+: Error-correcting code for Binary and DNA Data Storage applications},
	author = {Ramy Khabbaz},
	year = {2025},
	url = {https://github.com/ramy-khabbaz/mgcp},
	version = {1.1.0},
}
```

## License

MIT — see `LICENSE`.
