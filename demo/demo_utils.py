from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import os
import shutil
import subprocess
from pathlib import Path
import random
from collections import Counter
import tempfile
import traceback
import kalign
import psutil
from pycdhit import CDHIT, read_fasta
from tqdm import tqdm

def check_tool_installed(tool_name: str):
    """Ensure required binary (cd-hit) is installed."""
    if shutil.which(tool_name) is None:
        raise RuntimeError(
            f"Required tool '{tool_name}' not found in PATH.\n"
            f"Please install it before running."
        )

def txt_to_fasta(txt_path, fasta_path=None, prefix="read"):
    """
    Convert a text file (one DNA read per line) into a FASTA file for CD-HIT or other tools.

    Args:
        txt_path (str or Path): Path to the input .txt file (each line = one read).
        fasta_path (str or Path, optional): Output FASTA file path. 
                                            If None, saves as same name with .fasta extension.
        prefix (str): Prefix used to generate unique sequence headers (default: 'read').

    Returns:
        Path: Path to the generated FASTA file.
    """
    txt_path = Path(txt_path)
    if fasta_path is None:
        fasta_path = txt_path.with_suffix(".fasta")

    with txt_path.open("r", encoding="utf-8") as fin, fasta_path.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, start=1):
            seq = line.strip()
            if not seq:
                continue  # skip empty lines
            fout.write(f">{prefix}_{i}\n{seq}\n")

    return fasta_path

def run_cd_hit(input_txt, output_filename, identity=0.9, word_size=10, processes=cpu_count() // 2):
    """Run CD-HIT clustering on the given txt file."""
    check_tool_installed("cd-hit-est")

    fasta_file = txt_to_fasta(input_txt)
    cmd = [
        "cd-hit-est",
        "-i", fasta_file,
        "-o", output_filename,
        "-c", str(identity),
        "-n", str(word_size),
        "-T", str(processes)
    ]
    subprocess.run(cmd, check=True)

def consensus_from_aligned_strs(aligned_strs):
    """
    Build a consensus string from aligned sequences (with gaps).
    For each column, select the majority character.
    If the majority is a gap, the column is dropped.
    Ties are broken uniformly at random.
    """
    L = len(aligned_strs[0])
    consensus = []
    for i in range(L):
        col = [s[i] for s in aligned_strs]
        counts = Counter(col)
        max_ct = max(counts.values())
        top_bases = [b for b, ct in counts.items() if ct == max_ct]
        chosen_base = random.choice(top_bases)
        if chosen_base == '-':
            continue  # drop column if majority is gap
        consensus.append(chosen_base)
    return ''.join(consensus)

def get_consensus(copies):
    """
    Given a list of DNA sequence strings, align them with kalign
    and return the consensus sequence.

    """
    valid_copies = [s for s in copies if len(s) > 2]

    # Edge cases
    if not valid_copies:
        return ''
    if len(valid_copies) < 2 or all(s == valid_copies[0] for s in valid_copies):
        return valid_copies[0]
    
    # Attempt MSA-based consensus
    try:
        aligned = kalign.align(valid_copies)
        return consensus_from_aligned_strs(aligned)
    except Exception:
        traceback.print_exc()
        return ''

def cluster_cdhit(reads_or_fastq, identity=0.90, word_size=8, threads=None, memory=None, tmp_prefix=None):
    """
    Cluster reads using CD-HIT-EST (input can be a txt file with one read per line,
    a list of sequence strings, or a list of tuples (id, seq, ...)).
    Returns clusters keyed by CD-HIT cluster IDs, unassigned (empty here), and stats.
    """
    if threads is None:
        threads = cpu_count()
    if memory is None:
        memory = int(0.9 * (psutil.virtual_memory().total // (1024 * 1024)))

    tmp_dir = tmp_prefix or tempfile.mkdtemp()
    fasta_in = os.path.join(tmp_dir, "reads_for_cdhit.fasta")

    # Build fasta_in and read_map: header -> full read tuple
    read_map = {}

    # Helper to write a seq with given header to fasta
    def _write_fasta_records(records):
        # records: iterable of (header, seq)
        with open(fasta_in, "w", encoding="utf-8") as fout:
            for h, s in records:
                fout.write(f">{h}\n{s}\n")

    if isinstance(reads_or_fastq, list):
        # List of tuples (id, seq, ...) or list of sequence strings
        if reads_or_fastq and (isinstance(reads_or_fastq[0], (list, tuple)) and len(reads_or_fastq[0]) >= 2):
            records = []
            for r in reads_or_fastq:
                rid = str(r[0])
                seq = str(r[1])
                read_map[rid] = r
                records.append((rid, seq))
            _write_fasta_records(records)
        else:
            # list of plain sequences
            records = []
            for i, seq in enumerate(reads_or_fastq, start=1):
                rid = f"read_{i}"
                read_map[rid] = (rid, seq, None, None, None)
                records.append((rid, seq))
            _write_fasta_records(records)
    else:
        # Assume it is a path to a txt file (one read per line)
        txt_path = str(reads_or_fastq)
        # Use txt_to_fasta to create fasta_in (it will name headers read_1, read_2, ...)
        fasta_in = txt_to_fasta(txt_path, fasta_path=None, prefix="read")
        # Build read_map by reading the txt file
        with open(txt_path, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin, start=1):
                seq = line.strip()
                if not seq:
                    continue
                rid = f"read_{i}"
                read_map[rid] = (rid, seq, None, None, None)

    # Run CD-HIT-EST via pycdhit
    cdhit = CDHIT(prog="cd-hit-est")
    cdhit.set_options(c=identity, T=threads, M=memory, n=word_size, d=0)
    print("Clustering with CD-HIT-EST...")
    # Pass the fasta filename; pycdhit accepts a path or list depending on version
    df_rep, df_clstr = cdhit.cluster(read_fasta(fasta_in))

    # Build clusters keyed by CD-HIT cluster IDs
    clusters = {}
    for _, row in df_clstr.iterrows():
        cid = row["cluster"]
        rid = row["identifier"]
        if cid not in clusters:
            clusters[cid] = []
        # Only add if present in read_map (should be)
        if rid in read_map:
            clusters[cid].append(read_map[rid])
        else:
            # fallback: just store identifier
            clusters[cid].append((rid, None, None, None, None))

    # Stats
    total_reads = len(read_map)
    assigned_reads = sum(len(members) for members in clusters.values())
    unassigned = []  # CD-HIT typically assigns all reads into clusters
    assigned_by_barcode = Counter({cid: len(members) for cid, members in clusters.items()})
    unused_barcodes = set()

    stats = {
        "total_reads": total_reads,
        "assigned": assigned_reads,
        "unassigned": len(unassigned),
        "assigned_by_barcode": assigned_by_barcode,
        "unused_barcodes": unused_barcodes,
    }

    out_cluster_file = "clusters.txt"
    with open(out_cluster_file, "w") as out:
        for bc, members in clusters.items():
            out.write(f">cluster_{bc}\n")
            for (read_idx, payload, full_seq, qual, assigned) in members:
                out.write(f"{read_idx}\t{payload}\n")

    # save clustering stats
    with open("clustering_stats.txt", "w") as f:
        f.write(f"total_reads: {stats['total_reads']}\n")
        f.write(f"assigned: {stats['assigned']}\n")
        f.write(f"unassigned: {stats['unassigned']}\n")
        f.write("unused_barcodes:\n")
        for bc in stats["unused_barcodes"]:
            f.write(f"  {bc}\n")
        f.write("assigned_by_barcode:\n")
        for bc, count in stats["assigned_by_barcode"].most_common():
            f.write(f"  {bc}: {count}\n")

    return clusters, stats

def consensus_task_cdhit(payloads):
    """
    Compute consensus for a list of payloads, ignoring any None values.
    """
    valid_payloads = [p for p in payloads if p is not None]
    if valid_payloads:
        return get_consensus(valid_payloads)
    else:
        return "Failed"

def kalign_and_consensus(clusters, n_workers=None):
    """
    Compute consensus for CD-HIT clusters using kalign.
    """
    if n_workers is None:
        n_workers = cpu_count()

    # Step 1: compute consensus for each CD-HIT cluster in parallel
    cluster_payloads = []
    cluster_ids = []

    sorted_clusters = sorted(clusters.items(), key=lambda item: len(item[1]), reverse=True)
    for cid, members in sorted_clusters:
        cluster_ids.append(cid)
        # extract payloads for this cluster
        payloads = [payload for (read_idx, payload, full_seq, qual, assigned) in members]
        cluster_payloads.append(payloads)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        consensuses = list(
            tqdm(
                executor.map(consensus_task_cdhit, cluster_payloads),
                total=len(cluster_payloads),
                desc="Alignment and consensus",
            )
        )

    output_file = "consensuses.txt"
    with open(output_file, "w") as f:
        for cons in consensuses:
            if cons is not None:  # skip empty results
                f.write(cons + "\n")

    return consensuses