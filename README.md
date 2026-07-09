# MGCP — MGC+ coding for DNA and binary channels with insertion, deletion, and substitution (IDS) errors

MGCP is a Python package implementing the Marker Guess & Checl Plus (MGC+) family of encoders and decoders for both binary and DNA sequences. It contains:

- Encoders/decoders for binary and DNA sequences (`mgcp.binary`, `mgcp.dna`).
- File-level codec that encodes a binary file into a collection of DNA sequences and decodes it back from noisy DNA reads (`mgcp.dna.codec`).
- Utility modules for simulation, error models, and plotting (`mgcp.utils`).
- Command-line interface (`mgcp/cli`) that exposes the main workflows.
- Demos under `demo/` that show end-to-end examples (these require optional external tools).

## Table of contents

<p>
1. <a href="#1-installation-python--optional-system-deps">Installation</a><br>
2. <a href="#2-detailed-module-overview">Detailed module overview and examples</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;2.1 <a href="#21-mgcpbinary">mgcp.binary</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;2.2 <a href="#22-mgcpdna">mgcp.dna</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;2.3 <a href="#23-mgcpcodec">mgcp.codec</a><br>
3. <a href="#3-command-line-interface-and-examples">Command-line interface and examples</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;3.1 <a href="#41-mgcp-binary">mgcp binary</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;3.2 <a href="#42-mgcp-dna">mgcp dna</a><br>
&nbsp;&nbsp;&nbsp;&nbsp;3.3 <a href="#43-mgcp-codec">mgcp codec</a><br>
4. <a href="#4-multi-packet-support">Multi-packet support</a><br>
5. <a href="#5-demo--external-tools">Demo & external tools</a><br>
6. <a href="#6-citing-this-work">Citing this work</a><br>
7. <a href="#7-license">License</a>
</p>

## 1. Installation (Python + optional system deps)

Prerequisites:

- Python 3.10 or newer.
- A working C/Python toolchain only if you need to build optional native dependencies.

Clone and install locally:

```bash
git clone https://github.com/ramy-khabbaz/mgcp.git
cd mgcp
pip install -e .
```

Runtime dependencies are declared in `setup.cfg`: `numpy`, `scipy`, `tqdm`, `typer`, `reedsolo`, `galois`, and `matplotlib`.

### Optional demo/system dependencies

The core library and CLI do not require external bioinformatics tools. The full demo pipeline uses clustering and multiple sequence alignment tools, so install the demo extras when you want to run the end-to-end demo:

```bash
pip install -e '.[demo]'
```

Demo extras install:

- `kalign` - Python wrapper for Kalign MSA.
- `py-cdhit` - lightweight wrapper for CD-HIT.
- `psutil`, `tqdm`.

The native `cd-hit-est` and `kalign` executables may still need to be installed separately and available on `PATH`. See [Demo & external tools](#6-demo--external-tools).

## 2. Detailed module overview and examples

### 2.1 `mgcp.binary`

`mgcp.binary` is the binary module of the MGCP package. The encoder takes a binary message as input and outputs a binary codeword. The decoder takes a binary received sequence, which may contain IDS errors, as input and outputs either the decoded binary message or a decoding error flag. Use this module when you want to use MGC+ as a binary forward error-correcting code.

```python
from mgcp.binary import encode, decode

codeword, metadata = encode(
    binary_message="0101010011110110",
    l=4,
    parities_count=4,
    marker_period=2,
    export_json=True,
    export_path="mgcp_encode_meta.json",
)
print(codeword)

corrupted_sequence = codeword.copy()
# Substitute the 19th codeword bit (counting from 0)
corrupted_sequence[19] = 1 - corrupted_sequence[19]   # flips 0 to 1 or 1 to 0
# Delete the 5th and 6th codeword bits
del corrupted_sequence[5:7]
print(corrupted_sequence)

decoded_message, diagnostics = decode(
    corrupted_sequence,
    metadata=metadata,
    meta_path=None,
    Pd=None,
    Pi=None,
    Ps=None,
    return_diagnostics=True,
)
print(decoded_message)
```

#### Encoding parameters

- `binary_message`: Information bits, provided either as a binary string such as `"0101"` or as a list of `0` and `1` values such as `[0, 1, 0, 1]`. Its length must be divisible by `l`.
- `l`: Symbol length (block size) in bits. It must not exceed `8`.
- `parities_count`: Number of guess parities added by the MGC+ encoder. Each parity adds `l` redundant bits.
- `marker_period`: Marker mode. Use `0` for no marker, `1` for period-1 markers, and `2` for period-2 markers.
- `export_json`: (Optional; Default: `False`) When `True`, writes metadata to a local JSON file for later decoding.
- `export_path`: (Optional; Default: `mgcp_encode_meta.json` in the current directory) Path for the exported metadata JSON file.

For `marker_period=2`, the number of data blocks plus `parities_count` must be even.

#### Decoding parameters

- `corrupted_sequence`: Corrupted (noisy) binary sequence as a string, list, or other iterable of `0` and `1` values.
- `metadata`: Metadata returned by `encode(...)`.
- `meta_path`: (Optional) Path for metadata JSON. Used when `metadata` is not passed.
- `Pd`, `Pi`, `Ps`: (Optional) Channel deletion, insertion, and substitution probabilities. Provide all three together, or leave all unset.
- `return_diagnostics`: (Optional; Default: `False`) When `True`, returns decoding diagnostics.

When `Pd`, `Pi`, and `Ps` are not passed to the decoder, it estimates them from the `corrupted_sequence` if markers are included.

### 2.2 `mgcp.dna`

`mgcp.dna` is the DNA (quaternary) module of the MGCP package. The encoder takes a binary message as input and outputs a DNA codeword over `A`, `T`, `C`, and `G`. The decoder takes a DNA sequence, which may contain IDS errors, as input and outputs either the decoded binary message or a decoding error flag. Use this module when you want to use MGC+ as a quaternary (DNA) forward error-correcting code.

```python
from mgcp.dna import encode, decode

codeword, metadata = encode(
    binary_message="0101010011110110",
    l=4,
    parities_count=4,
    marker_period=0,
    export_json=True,
    export_path="mgcp_encode_meta.json",
)
print(codeword)
    
corrupted_sequence = list(codeword)
# Substitute the 1st base (counting from 0) from T to A
corrupted_sequence[1] = "A"
# Substitute the 16th base from C to T
corrupted_sequence[16] = "T"
# Delete the 9th and 10th bases
del corrupted_sequence[9:11]
# Insert G at the 3rd position
corrupted_sequence.insert(3, "G")
corrupted_sequence = "".join(corrupted_sequence)
print(corrupted_sequence)

decoded_message, diagnostics = decode(
    corrupted_sequence,
    metadata=metadata,
    meta_path=None,
    Pd=None,
    Pi=None,
    Ps=None,
    return_diagnostics=True,
)
print(decoded_message)
```

#### Encoding parameters

- `binary_message`: Information bits, provided either as a binary string such as `"0101"` or as a list of `0` and `1` values such as `[0, 1, 0, 1]`. Its length must be divisible by `l`.
- `l`: Symbol length (block size) in bits. It must not exceed `8`.
- `parities_count`: Number of guess parities added by the MGC+ encoder. Each parity adds `l` redundant bits.
- `marker_period`: Marker mode. Use `0` for no marker, `1` for period-1 markers, and `2` for period-2 markers.
- `export_json`: (Optional; Default: `False`) When `True`, writes metadata to a local JSON file for later decoding.
- `export_path`: (Optional; Default: `mgcp_encode_meta.json` in the current directory) Path for the exported metadata JSON file.

For `marker_period=2`, the number of data blocks plus `parities_count` must be even.

#### Decoding parameters

- `corrupted_sequence`: Corrupted (noisy) DNA sequence over `A`, `T`, `C`, and `G`.
- `metadata`: Metadata returned by `encode(...)`.
- `meta_path`: (Optional) Path of metadata JSON. Used when `metadata` is not passed.
- `Pd`, `Pi`, `Ps`: (Optional) Channel deletion, insertion, and substitution probabilities. Provide all three together, or leave all unset.
- `return_diagnostics`: (Optional; Default: `False`) When `True`, returns decoding diagnostics.

When `Pd`, `Pi`, and `Ps` are not passed to the decoder, it estimates them from the `corrupted_sequence` if markers are included.

### 2.3 `mgcp.codec`

`mgcp.codec` is the file-level codec module of the MGCP package. The encoder takes a file such as `.bin`, `.pdf`, `.jpg`, or `.txt` as input, unpacks it into binary bits, and outputs a collection of DNA sequences representing the file. The decoder takes noisy reads of the encoded DNA sequences that may contain IDS errors and outputs the reconstructed file or an error flag.

```python
from mgcp.dna.codec import encode, decode
from mgcp.utils.tools import generate_random_file

def main():
    
    generate_random_file(filename="data.bin", size_bytes=5 * 1024)

    encode(
        file_name="data.bin",
        max_length=120,
        inner_redundancy=4,
        outer_rate=0.9,
        input_path="./",
        useMarker=False,
        filtered=False,
        seed=123456,
        max_seqs_per_packet=None,
        processes=4,
    )

    details = decode(
        file_name="encoded_file.txt",
        input_path="./",
        processes=4,
        return_details=True,
    )
    
if __name__ == "__main__":
    main()
```

#### Encoding parameters

- `file_name`: Input file name, for example `data.bin`.
- `max_length`: Target maximum DNA sequence length. The encoder chooses the largest possible encoded sequence length below the specified maximum.
- `inner_redundancy`: Number of guess parities used by the inner DNA-MGC+ code. Set `0` to exclude inner redundancy.
- `outer_rate`: Outer code rate in `(0, 1)`. For example, `0.9` means about 90% information sequences and 10% redundant sequences.
- `input_path`: (Optional; Default: current directory) Folder path containing the input file.
- `useMarker`: (Optional; Default: `True`) Use period-2 marker-based inner encoding when `True`; use no-marker encoding when `False`.
- `filtered`: (Optional; Default: `False`) When `True`, each encoded packet contains the full `2^16 = 65536` candidate sequences, and the output reports how many sequences to retain to attain the specified `outer_rate`. The sequences to be retained should be selected manually by the user based on the desired content-specific constraints. The implementation does not support automatic filtering based on predefined constraints.
- `seed`: (Optional; Default: `123456`) Seed value to randomize the order of sequence and packet IDs.
- `max_seqs_per_packet`: (Optional; Default: `4096`) Optional upper bound on the number of encoded sequences per packet. Default: `4096`.
- `processes`: (Optional; Default: all detected CPU cores) Worker count for parallelized packet encoding.

Encoding writes:

- `encoded_file.txt`: DNA sequences, with visible `# PACKET ...` separators in filtered mode.
- `encoding_params.json`: Compact metadata with the `params` object needed to reconstruct the file during decoding.

#### Decoding parameters

- `file_name`: Text file containing DNA reads or consensus sequences, usually `encoded_file.txt` or `reads.txt`.
- `input_path`: (Optional; Default: current directory) folder path containing `file_name`.
- `processes`: (Optional; Default: all detected CPU cores) Worker count for parallelized inner and outer decoding.
- `return_details`: (Optional; Default: `False`) When `True`, returns decoding diagnostics.

Decoder reads `encoding_params.json` from the current working directory and writes `decoded_output<original_extension>`.

#### Hard-coded MGC+ code parameters

The following code parameters are hard-coded in the `mgcp.codec` module:

- Inner MGC+ symbol length (block size) in bits: `l_in = 8`.
- Outer Reed-Solomon symbol length in bits: `l_out = 16`.
- Sequence and packet index length in bits: `16` bits each.
- Max number of encoded sequences per packet: `2^l_out = 2^16 = 65536`.
- Marker period: `2` (if `useMarker = True`).

## 3. Command-line interface and examples

MGCP exposes one entrypoint, `mgcp`, with three main subcommands:

- `mgcp binary ...` for bit-level encoding, decoding, and plotting.
- `mgcp dna ...` for single DNA-sequence encoding, decoding, and plotting.
- `mgcp codec ...` for file-level encoding and decoding.

Use `--help` at any level; examples:

```bash
mgcp --help
mgcp binary encode --help
mgcp dna plot --help
mgcp codec encode --help
```

### 3.1 `mgcp binary`

Encode a single binary message into a binary codeword, with the symbol length (block size) set to 4 bits, 4 guess parities added, and the marker period set to 2:

```bash
mgcp binary encode "0101010011110110" 4 4 2
```

Decode the binary message from a corrupted sequence (the 5th and 6th bits of the codeword are deleted and the 19th is substituted):

```bash
mgcp binary decode "0101000011111011010111110111001010101100010010100101101100100011"
```

Generate a frame error rate vs. code rate (FER-vs-coderate) plot for the binary case:

```bash
mgcp binary plot fer-vs-coderate 140 7 2 6,8,10,12,14 --pe 0.01 --num-iterations 1000
```

Generate a frame error rate vs. channel error rate (FER-vs-Pe) plot for the binary case:

```bash
mgcp binary plot fer-vs-pe 140 7 8 2 --pe-min 0.001 --pe-max 0.011 --pe-step 0.001 --num-iterations 1000
```

### 3.2 `mgcp dna`

Encode a single binary message into a single DNA sequence, with the symbol length (block size) set to 4 bits, 4 guess parities added, and the marker period set to 0 (no markers):
```bash
mgcp dna encode "0101010011110110" 4 4 0
```

Decode the binary message from a corrupted DNA sequence (substitutions: 1st (T->A) and 16th (C->T) bases, deletions: 9th and 10th bases, insertion: 'G' is inserted at the 3rd position):

```bash
mgcp dna decode "TATGAGGTCGGTTTCTCTGATTGTGTT"
```

Generate a DNA frame error rate vs. code rate (FER-vs-coderate) plot:

```bash
mgcp dna plot fer-vs-coderate 256 8 2 6,8,10,12,14 --pe 0.01 --num-iterations 1000
```

Generate a DNA frame error rate vs. channel error rate (FER-vs-Pe) plot:

```bash
mgcp dna plot fer-vs-pe 256 8 8 2 --pe-min 0.001 --pe-max 0.011 --pe-step 0.001 --num-iterations 1000
```

### 3.3 `mgcp codec`

Encode any file into DNA sequences:

```bash
mgcp codec encode "data.bin" 120 4 0.9 --input-path ./ --no-marker
mgcp codec encode "document.pdf" 120 4 0.9 --input-path ./ --no-marker
mgcp codec encode "image.jpg" 120 4 0.9 --input-path ./ --no-marker
mgcp codec encode "notes.txt" 120 4 0.9 --input-path ./ --no-marker
```

Enable filtered output to generate all `2^16 = 65536` candidate DNA sequences and print how many sequences to retain to attain the desired outer rate:

```bash
mgcp codec encode "data.bin" 120 4 0.9 --input-path ./ --no-marker --filtered
```

Retrieve the file from noisy DNA reads or consensus sequences:

```bash
mgcp codec decode "encoded_file.txt" --input-path ./ --processes 4
mgcp codec decode "reads.txt" --input-path ./ --processes 4
```

The original file extension is restored automatically from `encoding_params.json`.

## 4. Multi-packet support

Large files are automatically split into independent packets during codec encoding. Multi-packet handling is transparent to the user: call `encode(...)` and `decode(...)` normally, and the codec handles packet planning, packet IDs, parallel decoding, and reconstruction.

**When are multiple packets used?** Automatically when the file would require more than `max_seqs_per_packet` (Default: `4096`) DNA sequences per packet. Smaller files use a single packet with no packet ID overhead.

### How it works

#### Encoding

- The input file is converted to bits and split into fixed-size chunks.
- Each packet is assigned a logical packet ID. Multi-packet files prepend a randomized 16-bit packet ID before each sequence ID.
- Each packet is independently inner-encoded and outer-encoded.
- Without filtering, each packet follows the requested `outer_rate`.
- With `filtered=True`, the codec keeps the same normal packet plan, generated the full `2^16 = 65536` candidate sequences for each packet, and prints how many sequences to retain to achieve the desired `outer_rate`.
- All DNA sequences are written to `encoded_file.txt`.

#### Decoding

- DNA sequences are read from the input text file. Empty lines and lines beginning with `#` are ignored.
- Inner decoding recovers packet IDs and sequence IDs.
- DNA sequences are grouped by packet ID.
- Each packet is outer-decoded independently, using parallel workers when available.
- Decoded packet payloads are concatenated in packet order.

The compact `encoding_params.json` stores the metadata needed to derive the packet plan during decode.

## 5. Demo & external tools

The `demo/` folder demonstrates the full file-level pipeline: encode a file, simulate noisy reads, cluster reads with CD-HIT, align clusters with Kalign, generate consensus sequences, and decode back.

The main demo, `demo/demo_dna_pipeline.py`, is intentionally written as a readable top-level script. Edit the parameters near the top of the file to change the file size, channel error rates, or output folder. Generated files are grouped under `output/` by default.

### External tools used by demos

- CD-HIT (`cd-hit-est`) - clustering. Install the executable and ensure `cd-hit-est` is on `PATH`. The demo extras install only Python wrappers; the native binary must be installed separately.
- Kalign - multiple sequence aligner. Install the Kalign binary or Kalign3 and ensure it is on `PATH`. The Python wrapper may still require the Kalign executable.

References and links:

- CD-HIT - Li, W. & Godzik, A. (2006). Cd-hit: a fast program for clustering and comparing large sets of protein or nucleotide sequences. Bioinformatics 22(13):1658-1659. DOI: https://doi.org/10.1093/bioinformatics/btl158. Project: https://github.com/weizhongli/cdhit
- Kalign - Lassmann, T. & Sonnhammer, E. L. L. (2005). Kalign, an accurate and fast multiple sequence alignment algorithm. BMC Bioinformatics 6:298. DOI: https://doi.org/10.1186/1471-2105-6-298. Kalign homepage: https://msa.sbc.su.se/kalign/ and Kalign3: https://github.com/TimoLassmann/kalign3

### Example demo outline

1. `mgcp.dna.codec.encode` generates `encoded_file.txt`.
2. `mgcp.utils.tools.error_generator` generates `reads.txt` with IDS errors.
3. CD-HIT clusters similar reads together.
4. Kalign aligns each cluster and produces `consensuses.txt`.
5. `mgcp.dna.codec.decode` decodes the consensus sequences to recover the file.
6. The demo compares `decoded_output.*` with the original file.

## 6. Citing this work

If you use MGCP in your research, please cite the following paper.

BibTeX example:

```bibtex
@article{mgcp,
  title   = {{DNA-MGC+}: A versatile codec for reliable and resource-efficient data storage on synthetic {DNA}},
  author  = {Khabbaz, Ramy and Mateos, J{\'e}r{\'e}my and Antonini, Marc and {Kas Hanna}, Serge},
  journal = {bioRxiv preprint,},
  publisher = {Cold Spring Harbor Laboratory},
  year    = {2026},
  doi     = {10.64898/2026.03.11.711016},
}
```

A machine-readable citation file is provided in `CITATION.cff`. It keeps the software metadata and sets this paper as the preferred citation.

## 7. License

MIT - see `LICENSE`.
