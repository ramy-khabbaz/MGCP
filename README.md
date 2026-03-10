# MGCP — MGC+ coding for DNA and binary channels

MGCP is a Python package implementing the MGC+ family of encoders and decoders for both binary and DNA data. It contains:

- Encoders/decoders for binary streams and DNA sequences (`mgcp.binary`, `mgcp.dna`).
- File-level codec that encodes binary files into DNA sequences and decodes them back from noisy reads (`mgcp.dna.codec`).
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

- Encode/decode at the bit level and the DNA level using MGC+ algorithms.
- Plotting helpers to benchmark FER vs coderate or error probability (both DNA and binary).
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

Output a detailed list of the DNA-MGC+ encoding parameters:

```bash
mgcp codec encode --help
```

Encode a single binary message, the block length is 8, 6 guess parities are added and the marker period is set to 2:

```bash
mgcp binary encode "0101010011110110" 8 6 2
```

Decode the binary message back (the 7th bit is deleted and the 20th is substituted):

```bash
mgcp binary decode 0101010111101100011001001110111010001011000110011001100111111111010011100011100100001001101100111
```

Encode a single binary message into a single DNA sequence:

```bash
mgcp dna encode "0101010011110110" 4 4 1
```

Decode the single DNA sequence back to binary (in this example the 4th and 14th nucliotides are deleted):

```bash
mgcp dna decode "TTACTAACGGACTCACGGACTGACTTACTCACCCTGATTGTGTT"
```

Encode a binary file to DNA sequences (file-level codec). Here, the file is encoded with a target oligo length of 120, the desired guess parities per encoded sequence is 4, markers were added and the set number of added redundant encoded sequences is 200:

```bash
mgcp codec encode data.bin 120 4 200 --input-path ./ --use-marker
```

Decode noisy DNA sequences (reads.txt) back to the file, 4 cores are used in parallel for decoding:

```bash
mgcp codec decode reads.txt --input-path ./ --processes 4
```

### Plotting

Both `mgcp dna` and `mgcp binary` include `plot` subcommands to generate FER vs coderate or FER vs Pe. Example:

```bash
mgcp dna plot fer-vs-coderate 1000 126 1 "1,2,3,4" --pe 0.01 --num-iterations 500
```

## Detailed module overview

- `mgcp.binary` — binary-level encoding/decoding primitives and utilities.
- `mgcp.dna` — binary input to DNA sequence encoding, decoding, and helper pipelines.
- `mgcp.dna.codec` — high-level file codec (binary file -> DNA sequences -> text) and reverse.
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
2. `mgcp.utils.tools.error_generator` to simulate reads with Pd/Pi/Ps errors.
3. Run CD-HIT on the reads to cluster similar reads together (`cd-hit-est` or the `pycdhit` helper).
4. For each cluster, run Kalign to MSA the reads and produce a consensus.
5. Feed consensus reads into `mgcp.dna.codec.decode` to recover the original file.

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
