"""
extract_features.py

Core functions for extracting features from cfDNA BAM files.
Handles fragment length, end motifs, and methylation analysis.

Features extracted:
- Aggregate statistics per sample
- Per-read metrics
- Position-level methylation for DMR analysis

Requirements:
    pip install pysam numpy pandas scipy

Usage:
    from extract_features import extract_features_from_bam, extract_methylation_by_position
    
    # Aggregate stats per sample
    features = extract_features_from_bam("sample.bam", region="chr21")
    
    # Position-level methylation for DMRs
    meth_df = extract_methylation_by_position("sample.bam", region="chr21")
"""

import pysam
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
import os


############################################################
# POSITION-LEVEL METHYLATION EXTRACTION (for DMR analysis) #
############################################################

def extract_methylation_by_position(
    bam_path: str,
    region: Optional[str] = None,
    min_mapq: int = 20,
    min_coverage: int = 1,
    context: str = 'CpG'
) -> pd.DataFrame:
    """
    Extract methylation data at each genomic position.
    
    This is the key function for DMR analysis - it provides methylation
    status at every cytosine position in the genome (or region).
    
    Args:
        bam_path: Path to BAM file
        region: Genomic region (e.g., 'chr21')
        min_mapq: Minimum mapping quality
        min_coverage: Minimum reads covering a position to include it
        context: Which cytosine context to extract ('CpG', 'CHG', 'CHH', or 'all')
    
    Returns:
        DataFrame with columns: chrom, position, meth_count, unmeth_count, total, meth_rate
    """
    bam = pysam.AlignmentFile(bam_path, "rb")
    
    # Dictionary to accumulate counts: position -> {'meth': count, 'unmeth': count}
    position_counts = defaultdict(lambda: {'meth': 0, 'unmeth': 0})
    
    # Define which characters to look for based on context
    if context == 'CpG':
        meth_char, unmeth_char = 'Z', 'z'
    elif context == 'CHG':
        meth_char, unmeth_char = 'X', 'x'
    elif context == 'CHH':
        meth_char, unmeth_char = 'H', 'h'
    elif context == 'all':
        meth_chars = {'Z', 'X', 'H'}
        unmeth_chars = {'z', 'x', 'h'}
    else:
        raise ValueError(f"Unknown context: {context}. Use 'CpG', 'CHG', 'CHH', or 'all'")
    
    reads = bam.fetch(region=region) if region else bam.fetch()
    
    for read in reads:
        # Quality filters
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if read.mapping_quality < min_mapq:
            continue
        if read.is_duplicate:
            continue
        
        # Need XM tag for methylation
        if not read.has_tag('XM'):
            continue
        
        xm = read.get_tag('XM')
        chrom = read.reference_name
        
        # Get aligned positions (handles insertions/deletions via CIGAR)
        # aligned_pairs gives tuples of (query_pos, reference_pos)
        aligned_pairs = read.get_aligned_pairs()
        
        # Iterate through XM string with corresponding reference positions
        for query_pos, ref_pos in aligned_pairs:
            # Skip if either position is None (insertion/deletion)
            if query_pos is None or ref_pos is None:
                continue
            
            # Make sure we don't go out of bounds on XM string
            if query_pos >= len(xm):
                continue
            
            xm_char = xm[query_pos]
            
            # Check if this is a methylation call we care about
            if context == 'all':
                if xm_char in meth_chars:
                    position_counts[(chrom, ref_pos)]['meth'] += 1
                elif xm_char in unmeth_chars:
                    position_counts[(chrom, ref_pos)]['unmeth'] += 1
            else:
                if xm_char == meth_char:
                    position_counts[(chrom, ref_pos)]['meth'] += 1
                elif xm_char == unmeth_char:
                    position_counts[(chrom, ref_pos)]['unmeth'] += 1
    
    bam.close()
    
    # Convert to DataFrame
    rows = []
    for (chrom, pos), counts in position_counts.items():
        total = counts['meth'] + counts['unmeth']
        if total >= min_coverage:
            rows.append({
                'chrom': chrom,
                'position': pos,
                'meth_count': counts['meth'],
                'unmeth_count': counts['unmeth'],
                'total': total,
                'meth_rate': counts['meth'] / total
            })
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(['chrom', 'position']).reset_index(drop=True)
    
    return df


def merge_methylation_across_samples(
    sample_meth_dfs: Dict[str, pd.DataFrame],
    min_samples: int = 1
) -> pd.DataFrame:
    """
    Merge position-level methylation data across multiple samples.
    
    Args:
        sample_meth_dfs: Dict mapping sample_id -> methylation DataFrame
        min_samples: Minimum number of samples with data at a position to include it
    
    Returns:
        Wide-format DataFrame with one row per position, columns for each sample's meth_rate
    """
    # Collect all positions
    all_positions = set()
    for df in sample_meth_dfs.values():
        positions = set(zip(df['chrom'], df['position']))
        all_positions.update(positions)
    
    # Build merged dataframe
    rows = []
    for chrom, pos in sorted(all_positions):
        row = {'chrom': chrom, 'position': pos}
        
        sample_count = 0
        for sample_id, df in sample_meth_dfs.items():
            match = df[(df['chrom'] == chrom) & (df['position'] == pos)]
            if len(match) > 0:
                row[f'{sample_id}_meth_rate'] = match.iloc[0]['meth_rate']
                row[f'{sample_id}_coverage'] = match.iloc[0]['total']
                sample_count += 1
            else:
                row[f'{sample_id}_meth_rate'] = np.nan
                row[f'{sample_id}_coverage'] = 0
        
        row['n_samples'] = sample_count
        
        if sample_count >= min_samples:
            rows.append(row)
    
    return pd.DataFrame(rows)


#########################################
# PER-READ SYNCHRONIZED DATA EXTRACTION #
#########################################

def extract_per_read_data(
    bam_path: str,
    region: Optional[str] = None,
    min_mapq: int = 20,
    max_reads: Optional[int] = None
) -> pd.DataFrame:
    """
    Extract synchronized per-read data for correlation analysis.
    
    Unlike extract_features_from_bam() which returns aggregate statistics,
    this returns one row per read with all features aligned.
    
    Args:
        bam_path: Path to BAM file
        region: Genomic region (e.g., 'chr21')
        min_mapq: Minimum mapping quality
        max_reads: Maximum reads to process (for testing)
    
    Returns:
        DataFrame with one row per read, columns for all features
    """
    bam = pysam.AlignmentFile(bam_path, "rb")
    
    reads_data = []
    reads = bam.fetch(region=region) if region else bam.fetch()
    read_count = 0
    
    for read in reads:
        if max_reads and read_count >= max_reads:
            break
        
        # Quality filters
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if read.mapping_quality < min_mapq:
            continue
        if read.is_duplicate:
            continue
        if read.is_paired and read.is_read2:
            continue
        
        read_count += 1
        
        # Initialize row with None for all fields
        row = {
            'read_name': read.query_name,
            'chrom': read.reference_name,
            'start_pos': read.reference_start,
            'end_pos': read.reference_end,
            'mapq': read.mapping_quality,
            'fragment_length': None,
            'start_motif_4mer': None,
            'end_motif_4mer': None,
            'start_motif_2mer': None,
            'end_motif_2mer': None,
            'gc_content': None,
            'meth_rate': None,
            'cpg_meth': None,
            'cpg_unmeth': None,
            'chg_meth': None,
            'chg_unmeth': None,
            'chh_meth': None,
            'chh_unmeth': None,
        }
        
        # Fragment length
        if read.is_proper_pair:
            tlen = abs(read.template_length)
            if 0 < tlen < 1000:
                row['fragment_length'] = tlen
        
        # Sequence-based features
        seq = read.query_sequence
        if seq:
            # GC content of whole read
            gc = sum(1 for b in seq.upper() if b in 'GC')
            row['gc_content'] = gc / len(seq)
            
            # End motifs
            if len(seq) >= 4:
                start_4mer = seq[:4].upper()
                end_4mer = seq[-4:].upper()
                
                if 'N' not in start_4mer:
                    row['start_motif_4mer'] = start_4mer
                    row['start_motif_2mer'] = start_4mer[:2]
                if 'N' not in end_4mer:
                    row['end_motif_4mer'] = end_4mer
                    row['end_motif_2mer'] = end_4mer[:2]
        
        # Methylation
        if read.has_tag('XM'):
            xm = read.get_tag('XM')
            
            cpg_m = xm.count('Z')
            cpg_u = xm.count('z')
            chg_m = xm.count('X')
            chg_u = xm.count('x')
            chh_m = xm.count('H')
            chh_u = xm.count('h')
            
            row['cpg_meth'] = cpg_m
            row['cpg_unmeth'] = cpg_u
            row['chg_meth'] = chg_m
            row['chg_unmeth'] = chg_u
            row['chh_meth'] = chh_m
            row['chh_unmeth'] = chh_u
            
            total_c = cpg_m + cpg_u + chg_m + chg_u + chh_m + chh_u
            total_meth = cpg_m + chg_m + chh_m
            
            if total_c > 0:
                row['meth_rate'] = total_meth / total_c
        
        reads_data.append(row)
    
    bam.close()
    
    return pd.DataFrame(reads_data)


################################
# AGGREGATE FEATURE EXTRACTION #
################################

def aggregate_features_from_bam(
    bam_path: str,
    region: Optional[str] = None,
    min_mapq: int = 20,
    max_reads: Optional[int] = None
) -> Dict:
    """
    Extract aggregate cfDNA features from a BAM file.
    
    This is the main function for sample-level feature extraction.
    For per-read data, use extract_per_read_data().
    For position-level methylation (DMRs), use extract_methylation_by_position().
    
    Args:
        bam_path: Path to BAM file
        region: Genomic region to analyze (e.g., 'chr21'). None for all.
        min_mapq: Minimum mapping quality
        max_reads: Maximum reads to process (for testing). None for all.
    
    Returns:
        Dictionary containing all extracted features and raw data
    """
    bam = pysam.AlignmentFile(bam_path, "rb")
    
    # Data collectors
    fragment_lengths = []
    start_motifs_4mer = []
    end_motifs_4mer = []
    start_positions = []
    end_positions = []
    
    # Methylation counters (from Bismark XM tag)
    meth_counts = {
        'CpG': {'meth': 0, 'unmeth': 0},
        'CHG': {'meth': 0, 'unmeth': 0},
        'CHH': {'meth': 0, 'unmeth': 0}
    }
    per_fragment_methylation = []
    
    reads = bam.fetch(region=region) if region else bam.fetch()
    read_count = 0
    
    for read in reads:
        if max_reads and read_count >= max_reads:
            break
        
        # Quality filters
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if read.mapping_quality < min_mapq:
            continue
        if read.is_duplicate:
            continue
        if read.is_paired and read.is_read2:
            continue
        
        read_count += 1
        
        # Fragment length
        if read.is_proper_pair:
            tlen = abs(read.template_length)
            if 0 < tlen < 1000:
                fragment_lengths.append(tlen)
        
        # Positions
        start_positions.append(read.reference_start)
        if read.reference_end:
            end_positions.append(read.reference_end)
        
        # End motifs
        # collect every unique 4mer at start and end
        seq = read.query_sequence
        if seq and len(seq) >= 4:
            start_4mer = seq[:4].upper()
            end_4mer = seq[-4:].upper()
            
            if 'N' not in start_4mer:
                start_motifs_4mer.append(start_4mer)
            if 'N' not in end_4mer:
                end_motifs_4mer.append(end_4mer)
        
        # Methylation from XM tag
        if read.has_tag('XM'):
            xm = read.get_tag('XM')
            frag_meth, frag_total = _parse_xm_tag(xm, meth_counts)
            if frag_total > 0:
                per_fragment_methylation.append(frag_meth / frag_total)
    
    bam.close()
    
    # Calculate all features
    features = {
        'total_reads': read_count,
        'total_fragments': len(fragment_lengths),
    }
    
    features.update(_calc_fragment_features(fragment_lengths))
    features.update(_calc_motif_features(start_motifs_4mer, end_motifs_4mer))
    features.update(_calc_methylation_features(meth_counts, per_fragment_methylation))
    features.update(_calc_position_features(start_positions, end_positions))
    
    # Store raw data
    features['_raw'] = {
        'fragment_lengths': fragment_lengths,
        'start_motifs': start_motifs_4mer,
        'end_motifs': end_motifs_4mer,
        'start_positions': start_positions,
        'end_positions': end_positions,
        'per_fragment_methylation': per_fragment_methylation
    }
    
    return features


####################
# HELPER FUNCTIONS #
####################

def _parse_xm_tag(xm_string: str, meth_counts: Dict) -> Tuple[int, int]:
    """
    Parse Bismark XM methylation tag.
    
    XM codes:
        Z = methylated CpG,   z = unmethylated CpG
        X = methylated CHG,   x = unmethylated CHG
        H = methylated CHH,   h = unmethylated CHH
        . = not a cytosine
    
    Returns:
        (methylated_count, total_cytosines) for this read
    """
    meth = 0
    total = 0
    
    for char in xm_string:
        if char == 'Z':
            meth_counts['CpG']['meth'] += 1
            meth += 1
            total += 1
        elif char == 'z':
            meth_counts['CpG']['unmeth'] += 1
            total += 1
        elif char == 'X':
            meth_counts['CHG']['meth'] += 1
            meth += 1
            total += 1
        elif char == 'x':
            meth_counts['CHG']['unmeth'] += 1
            total += 1
        elif char == 'H':
            meth_counts['CHH']['meth'] += 1
            meth += 1
            total += 1
        elif char == 'h':
            meth_counts['CHH']['unmeth'] += 1
            total += 1
    
    return meth, total


def _calc_fragment_features(lengths: List[int]) -> Dict:
    """Calculate fragment length statistics."""
    if not lengths:
        return {
            'frag_mean': 0, 'frag_median': 0, 'frag_std': 0,
            'frag_q25': 0, 'frag_q75': 0, 'frag_iqr': 0,
            'frag_mode': 0, 'frag_min': 0, 'frag_max': 0,
            'frag_skewness': 0, 'frag_kurtosis': 0,
            'frag_ratio_short': 0, 'frag_ratio_mono': 0, 'frag_ratio_di': 0,
        }
    
    arr = np.array(lengths)
    q25, q75 = np.percentile(arr, [25, 75])
    
    counts = Counter(lengths)
    mode = counts.most_common(1)[0][0]
    
    from scipy import stats as scipy_stats
    skewness = scipy_stats.skew(arr) if len(arr) > 2 else 0
    kurtosis = scipy_stats.kurtosis(arr) if len(arr) > 3 else 0
    
    n = len(arr)
    short = np.sum(arr < 150) / n
    mono = np.sum((arr >= 150) & (arr < 250)) / n
    di = np.sum((arr >= 250) & (arr < 450)) / n
    
    return {
        'frag_mean': float(np.mean(arr)),
        'frag_median': float(np.median(arr)),
        'frag_std': float(np.std(arr)),
        'frag_q25': float(q25),
        'frag_q75': float(q75),
        'frag_iqr': float(q75 - q25),
        'frag_mode': int(mode),
        'frag_min': int(np.min(arr)),
        'frag_max': int(np.max(arr)),
        'frag_skewness': float(skewness),
        'frag_kurtosis': float(kurtosis),
        'frag_ratio_short': float(short),
        'frag_ratio_mono': float(mono),
        'frag_ratio_di': float(di),
    }


def _calc_motif_features(start_motifs: List[str], end_motifs: List[str]) -> Dict:
    """Calculate end motif statistics."""
    features = {}
    all_motifs = start_motifs + end_motifs
    
    if not all_motifs:
        features['motif_4mer'] = {}
        features['motif_diversity'] = 0
        features['motif_gc_content'] = 0
        features['motif_start_end_corr'] = 0
        return features
    
    # 4-mer frequencies
    motif_counts = Counter(all_motifs)
    total = len(all_motifs)
    
    features['motif_4mer'] = {motif: count / total for motif, count in motif_counts.items()}
    
    # GC content
    gc_count = sum(1 for m in all_motifs for b in m if b in 'GC')
    features['motif_gc_content'] = gc_count / (len(all_motifs) * 4)
    
    # Shannon diversity
    probs = np.array([c / total for c in motif_counts.values()])
    probs = probs[probs > 0]
    features['motif_diversity'] = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0
    
    # Start/End correlation
    start_counts = Counter(start_motifs)
    end_counts = Counter(end_motifs)
    all_unique = set(start_counts.keys()) | set(end_counts.keys())
    
    if len(start_motifs) > 0 and len(end_motifs) > 0:
        start_freq = np.array([start_counts.get(m, 0) / len(start_motifs) for m in all_unique])
        end_freq = np.array([end_counts.get(m, 0) / len(end_motifs) for m in all_unique])
        if np.std(start_freq) > 0 and np.std(end_freq) > 0:
            features['motif_start_end_corr'] = float(np.corrcoef(start_freq, end_freq)[0, 1])
        else:
            features['motif_start_end_corr'] = 1.0
    else:
        features['motif_start_end_corr'] = 0
    
    return features


def _calc_methylation_features(meth_counts: Dict, per_frag_meth: List[float]) -> Dict:
    """Calculate methylation statistics."""
    features = {}
    
    cpg_total = meth_counts['CpG']['meth'] + meth_counts['CpG']['unmeth']
    features['meth_cpg_rate'] = meth_counts['CpG']['meth'] / cpg_total if cpg_total > 0 else 0
    features['meth_cpg_sites'] = cpg_total
    
    chg_total = meth_counts['CHG']['meth'] + meth_counts['CHG']['unmeth']
    features['meth_chg_rate'] = meth_counts['CHG']['meth'] / chg_total if chg_total > 0 else 0
    features['meth_chg_sites'] = chg_total
    
    chh_total = meth_counts['CHH']['meth'] + meth_counts['CHH']['unmeth']
    features['meth_chh_rate'] = meth_counts['CHH']['meth'] / chh_total if chh_total > 0 else 0
    features['meth_chh_sites'] = chh_total
    
    total_meth = sum(d['meth'] for d in meth_counts.values())
    total_sites = cpg_total + chg_total + chh_total
    features['meth_global_rate'] = total_meth / total_sites if total_sites > 0 else 0
    features['meth_total_sites'] = total_sites
    
    if per_frag_meth:
        arr = np.array(per_frag_meth)
        features['meth_frag_mean'] = float(np.mean(arr))
        features['meth_frag_std'] = float(np.std(arr))
        features['meth_frag_median'] = float(np.median(arr))
    else:
        features['meth_frag_mean'] = 0
        features['meth_frag_std'] = 0
        features['meth_frag_median'] = 0
    
    return features


def _calc_position_features(start_pos: List[int], end_pos: List[int]) -> Dict:
    """Calculate position distribution statistics."""
    features = {}
    
    if start_pos:
        pos_counts = Counter(start_pos)
        counts = np.array(list(pos_counts.values()))
        cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0
        features['pos_coverage_cv'] = float(cv)
        features['pos_span'] = max(start_pos) - min(start_pos)
    else:
        features['pos_coverage_cv'] = 0
        features['pos_span'] = 0
    
    return features

##############################
# BATCH PROCESSING UTILITIES #
##############################

def batch_extract_features(
    bam_files: List[str],
    labels: List[str],
    region: Optional[str] = None,
    max_reads: Optional[int] = None,
    verbose: bool = True
) -> Tuple[List[Dict], List[str], List[str]]:
    """
    Extract aggregate features from multiple BAM files.
    """
    all_features = []
    sample_ids = []
    
    for i, (bam_path, label) in enumerate(zip(bam_files, labels)):
        sample_id = os.path.basename(bam_path).replace('.bam', '')
        
        if verbose:
            print(f"[{i+1}/{len(bam_files)}] Processing {sample_id} ({label})...")
        
        features = extract_features_from_bam(bam_path, region=region, max_reads=max_reads)
        features['label'] = label
        features['sample_id'] = sample_id
        
        all_features.append(features)
        sample_ids.append(sample_id)
        
        if verbose:
            print(f"    Fragments: {features['total_fragments']:,}")
            print(f"    Mean length: {features['frag_mean']:.1f} bp")
            print(f"    CpG methylation: {features['meth_cpg_rate']:.3f}")
    
    return all_features, sample_ids, labels


def batch_extract_methylation_by_position(
    bam_files: List[str],
    sample_ids: List[str],
    region: Optional[str] = None,
    min_coverage: int = 5,
    context: str = 'CpG',
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Extract position-level methylation from multiple BAM files.
    
    Args:
        bam_files: List of BAM file paths
        sample_ids: List of sample identifiers
        region: Genomic region
        min_coverage: Minimum coverage at a position
        context: Methylation context ('CpG', 'CHG', 'CHH', 'all')
        verbose: Print progress
    
    Returns:
        Dictionary mapping sample_id -> methylation DataFrame
    """
    sample_meth = {}
    
    for i, (bam_path, sample_id) in enumerate(zip(bam_files, sample_ids)):
        if verbose:
            print(f"[{i+1}/{len(bam_files)}] Extracting methylation for {sample_id}...")
        
        meth_df = extract_methylation_by_position(
            bam_path, 
            region=region, 
            min_coverage=min_coverage,
            context=context
        )
        
        sample_meth[sample_id] = meth_df
        
        if verbose:
            print(f"    Positions with data: {len(meth_df):,}")
            if len(meth_df) > 0:
                print(f"    Mean methylation: {meth_df['meth_rate'].mean():.3f}")
    
    return sample_meth


def features_to_dataframe(features_list: List[Dict], exclude_raw: bool = True) -> pd.DataFrame:
    """Convert list of feature dictionaries to pandas DataFrame."""
    if exclude_raw:
        clean_features = [{k: v for k, v in f.items() if k != '_raw'} for f in features_list]
        return pd.DataFrame(clean_features)
    return pd.DataFrame(features_list)