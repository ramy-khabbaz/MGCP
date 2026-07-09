from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import os
import shutil
import subprocess
from pathlib import Path
import random
from collections import Counter
import tempfile
import traceback
import psutil
from tqdm import tqdm
from pycdhit import CDHIT, read_fasta

if os.name != 'nt':
    import kalign

def check_tool_installed(tool_name: str):
    if shutil.which(tool_name) is None:
        raise RuntimeError(f"Required tool '{tool_name}' not found in PATH.")

def txt_to_fasta(txt_path, fasta_path=None, prefix="read"):
    txt_path = Path(txt_path)
    if fasta_path is None:
        fasta_path = txt_path.with_suffix(".fasta")
    with txt_path.open("r", encoding="utf-8") as fin, fasta_path.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, start=1):
            seq = line.strip()
            if not seq:
                continue
            fout.write(f">{prefix}_{i}\n{seq}\n")
    return fasta_path

def run_cd_hit(input_txt, output_filename, identity=0.9, word_size=10, processes=cpu_count() // 2):
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
    L = len(aligned_strs[0])
    consensus = []
    for i in range(L):
        col = [s[i] for s in aligned_strs]
        counts = Counter(col)
        max_ct = max(counts.values())
        top_bases = [b for b, ct in counts.items() if ct == max_ct]
        chosen_base = random.choice(top_bases)
        if chosen_base == '-':
            continue
        consensus.append(chosen_base)
    return ''.join(consensus)

def get_consensus(copies):
    valid_copies = [s for s in copies if len(s) > 2]
    if not valid_copies:
        return ''
    if len(valid_copies) < 2 or all(s == valid_copies[0] for s in valid_copies):
        return valid_copies[0]
    try:
        aligned = kalign.align(valid_copies)
        return consensus_from_aligned_strs(aligned)
    except Exception:
        traceback.print_exc()
        return ''

def cluster_cdhit(reads_or_fastq, identity=0.90, word_size=8, threads=None, memory=None, tmp_prefix=None):
    if threads is None:
        threads = cpu_count()
    if memory is None:
        memory = int(0.9 * (psutil.virtual_memory().total // (1024 * 1024)))

    tmp_dir = tmp_prefix or tempfile.mkdtemp()
    fasta_in = os.path.join(tmp_dir, "reads_for_cdhit.fasta")

    read_map = {}

    def _write_fasta_records(records):
        with open(fasta_in, "w", encoding="utf-8") as fout:
            for h, s in records:
                fout.write(f">{h}\n{s}\n")

    if isinstance(reads_or_fastq, list):
        if reads_or_fastq and (isinstance(reads_or_fastq[0], (list, tuple)) and len(reads_or_fastq[0]) >= 2):
            records = []
            for r in reads_or_fastq:
                rid = str(r[0])
                seq = str(r[1])
                read_map[rid] = r
                records.append((rid, seq))
            _write_fasta_records(records)
        else:
            records = []
            for i, seq in enumerate(reads_or_fastq, start=1):
                rid = f"read_{i}"
                read_map[rid] = (rid, seq, None, None, None)
                records.append((rid, seq))
            _write_fasta_records(records)
    else:
        txt_path = str(reads_or_fastq)
        fasta_in = txt_to_fasta(txt_path, fasta_path=None, prefix="read")
        with open(txt_path, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin, start=1):
                seq = line.strip()
                if not seq:
                    continue
                rid = f"read_{i}"
                read_map[rid] = (rid, seq, None, None, None)

    cdhit = CDHIT(prog="cd-hit-est")
    cdhit.set_options(c=identity, T=threads, M=memory, n=word_size, d=0)
    print("Clustering with CD-HIT-EST...")
    df_rep, df_clstr = cdhit.cluster(read_fasta(fasta_in))

    clusters = {}
    for _, row in df_clstr.iterrows():
        cid = row["cluster"]
        rid = row["identifier"]
        if cid not in clusters:
            clusters[cid] = []
        if rid in read_map:
            clusters[cid].append(read_map[rid])
        else:
            clusters[cid].append((rid, None, None, None, None))

    total_reads = len(read_map)
    assigned_reads = sum(len(members) for members in clusters.values())
    unassigned = []
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
    valid_payloads = [p for p in payloads if p is not None]
    if valid_payloads:
        return get_consensus(valid_payloads)
    else:
        return "Failed"


def kalign_and_consensus(clusters, n_workers=None):
    """
    Compute consensus for CD-HIT clusters using kalign.

    Parallelizes over clusters using ProcessPoolExecutor. Each worker
    handles one cluster at a time (serial within cluster, parallel across
    clusters). chunksize batches multiple clusters per IPC round-trip so
    the ~17800 small tasks don't each pay full fork/pickle overhead.
    """
    if n_workers is None:
        n_workers = cpu_count()

    sorted_clusters = sorted(clusters.items(), key=lambda item: len(item[1]), reverse=True)
    cluster_payloads = [
        [payload for (read_idx, payload, full_seq, qual, assigned) in members]
        for cid, members in sorted_clusters
    ]

    # Batch clusters into groups so each worker handles multiple clusters
    # per IPC round-trip. Without this, 17800 clusters = 17800 separate
    # pickle/unpickle cycles, and the overhead dominates the <1ms kalign calls.
    chunksize = max(1, len(cluster_payloads) // (n_workers * 4))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        consensuses = list(
            tqdm(
                executor.map(consensus_task_cdhit, cluster_payloads, chunksize=chunksize),
                total=len(cluster_payloads),
                desc="Alignment and consensus",
                disable=os.environ.get("TQDM_DISABLE") == "1"
            )
        )

    output_file = "consensuses.txt"
    with open(output_file, "w") as f:
        for cons in consensuses:
            if cons is not None:
                f.write(cons + "\n")

    return consensuses
