"""
cfdna_methylkit.py

Integration with methylKit for DMR analysis from Python.

This module provides:
1. Export functions to create methylKit-compatible files from BAM
2. rpy2 wrapper to run methylKit from Python
3. Functions to import methylKit results back to pandas

Requirements:
    pip install rpy2 pandas numpy
    
    In R:
    install.packages("BiocManager")
    BiocManager::install("methylKit")

Usage:
    from cfdna_methylkit import run_methylkit_dmr_analysis
    
    dmrs = run_methylkit_dmr_analysis(
        bam_files=bam_files,
        sample_ids=sample_ids,
        treatments=[1,1,1,0,0,0],  # 1=ALS, 0=CTRL
        output_dir="./methylkit_output"
    )
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path


#############################################
# EXPORT FUNCTIONS: BAM -> methylKit format #
#############################################

def export_bam_to_methylkit_format(
    bam_path: str,
    output_path: str,
    sample_id: str,
    region: Optional[str] = None,
    min_mapq: int = 20,
    context: str = 'CpG'
) -> str:
    """
    Extract methylation from BAM and export to methylKit text format.
    
    methylKit text format (tab-separated):
    chrBase    chr    base    strand    coverage    freqC    freqT
    chr21.100  chr21  100     F         10          80.00    20.00
    
    Args:
        bam_path: Path to BAM file with Bismark XM tags
        output_path: Output file path
        sample_id: Sample identifier
        region: Genomic region (e.g., 'chr21')
        min_mapq: Minimum mapping quality
        context: Methylation context ('CpG', 'CHG', 'CHH')
    
    Returns:
        Path to output file
    """
    import pysam
    from collections import defaultdict
    
    bam = pysam.AlignmentFile(bam_path, "rb")
    
    # Accumulate counts per position
    # Key: (chrom, position, strand) -> {'meth': count, 'unmeth': count}
    position_counts = defaultdict(lambda: {'meth': 0, 'unmeth': 0})
    
    # Define methylation characters
    if context == 'CpG':
        meth_char, unmeth_char = 'Z', 'z'
    elif context == 'CHG':
        meth_char, unmeth_char = 'X', 'x'
    elif context == 'CHH':
        meth_char, unmeth_char = 'H', 'h'
    else:
        raise ValueError(f"Unknown context: {context}")
    
    reads = bam.fetch(region=region) if region else bam.fetch()
    
    for read in reads:
        # Quality filters
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if read.mapping_quality < min_mapq:
            continue
        if read.is_duplicate:
            continue
        if not read.has_tag('XM'):
            continue
        
        xm = read.get_tag('XM')
        chrom = read.reference_name
        
        # Determine strand from Bismark XG tag if available, else from read
        if read.has_tag('XG'):
            # XG tag: CT for original top strand, GA for original bottom strand
            xg = read.get_tag('XG')
            strand = 'F' if xg == 'CT' else 'R'
        else:
            strand = 'F' if not read.is_reverse else 'R'
        
        # Get aligned positions
        aligned_pairs = read.get_aligned_pairs()
        
        for query_pos, ref_pos in aligned_pairs:
            if query_pos is None or ref_pos is None:
                continue
            if query_pos >= len(xm):
                continue
            
            xm_char = xm[query_pos]
            
            if xm_char == meth_char:
                position_counts[(chrom, ref_pos, strand)]['meth'] += 1
            elif xm_char == unmeth_char:
                position_counts[(chrom, ref_pos, strand)]['unmeth'] += 1
    
    bam.close()
    
    # Write to methylKit format
    with open(output_path, 'w') as f:
        # Header
        f.write("chrBase\tchr\tbase\tstrand\tcoverage\tfreqC\tfreqT\n")
        
        # Sort by position
        sorted_positions = sorted(position_counts.keys())
        
        for chrom, pos, strand in sorted_positions:
            counts = position_counts[(chrom, pos, strand)]
            total = counts['meth'] + counts['unmeth']
            
            if total > 0:
                freq_c = (counts['meth'] / total) * 100
                freq_t = (counts['unmeth'] / total) * 100
                
                chr_base = f"{chrom}.{pos}"
                f.write(f"{chr_base}\t{chrom}\t{pos}\t{strand}\t{total}\t{freq_c:.2f}\t{freq_t:.2f}\n")
    
    print(f"Exported {len(position_counts)} positions to {output_path}")
    return output_path


def export_bam_to_bismark_coverage(
    bam_path: str,
    output_path: str,
    region: Optional[str] = None,
    min_mapq: int = 20,
    context: str = 'CpG'
) -> str:
    """
    Extract methylation from BAM and export to Bismark coverage format.
    
    Bismark coverage format (tab-separated, no header):
    chr21   10000   10000   80.0    8   2
    chrom   start   end     meth%   meth_count  unmeth_count
    
    This format can be read directly by methylKit::methRead() with pipeline="bismarkCoverage"
    
    Args:
        bam_path: Path to BAM file
        output_path: Output file path
        region: Genomic region
        min_mapq: Minimum mapping quality
        context: Methylation context
    
    Returns:
        Path to output file
    """
    import pysam
    from collections import defaultdict
    
    bam = pysam.AlignmentFile(bam_path, "rb")
    position_counts = defaultdict(lambda: {'meth': 0, 'unmeth': 0})
    
    if context == 'CpG':
        meth_char, unmeth_char = 'Z', 'z'
    elif context == 'CHG':
        meth_char, unmeth_char = 'X', 'x'
    elif context == 'CHH':
        meth_char, unmeth_char = 'H', 'h'
    else:
        raise ValueError(f"Unknown context: {context}")
    
    reads = bam.fetch(region=region) if region else bam.fetch()
    
    for read in reads:
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if read.mapping_quality < min_mapq:
            continue
        if read.is_duplicate:
            continue
        if not read.has_tag('XM'):
            continue
        
        xm = read.get_tag('XM')
        chrom = read.reference_name
        aligned_pairs = read.get_aligned_pairs()
        
        for query_pos, ref_pos in aligned_pairs:
            if query_pos is None or ref_pos is None:
                continue
            if query_pos >= len(xm):
                continue
            
            xm_char = xm[query_pos]
            
            if xm_char == meth_char:
                position_counts[(chrom, ref_pos)]['meth'] += 1
            elif xm_char == unmeth_char:
                position_counts[(chrom, ref_pos)]['unmeth'] += 1
    
    bam.close()
    
    # Write to Bismark coverage format
    with open(output_path, 'w') as f:
        for (chrom, pos), counts in sorted(position_counts.items()):
            total = counts['meth'] + counts['unmeth']
            if total > 0:
                meth_pct = (counts['meth'] / total) * 100
                # Bismark coverage: chrom, start, end, meth%, meth_count, unmeth_count
                f.write(f"{chrom}\t{pos}\t{pos}\t{meth_pct:.1f}\t{counts['meth']}\t{counts['unmeth']}\n")
    
    print(f"Exported {len(position_counts)} positions to {output_path}")
    return output_path


###########################################
# RPY2 WRAPPER: Run methylKit from Python #
###########################################

def run_methylkit_tiled_dmr_analysis(
    coverage_files: list,
    sample_ids: list,
    treatments: list,
    output_dir: str = "./methylkit_results",
    min_per_group: int = 1,
    tile_window: int = 5000,
    tile_step: int = 5000,
    tile_min_coverage: int = 5,
    min_diff: float = 0.1,
    qvalue: float = 0.05,
    mc_cores: int = 1,
    verbose: bool = True
):
    """
    Run methylKit DMR analysis from coverage files using regional aggregation.
    
    Designed for low-coverage data - aggregates CpGs into windows before analysis.
    
    Args:
        coverage_files: List of paths to Bismark coverage files
        sample_ids: List of sample IDs
        treatments: List of treatment values (1 for ALS, 0 for Control)
        output_dir: Directory for output files
        min_per_group: Minimum samples per group with coverage at a position
        tile_window: Window size for regional aggregation (bp)
        tile_step: Step size for windows (bp)
        tile_min_coverage: Minimum total coverage per tile across pooled CpGs
        min_diff: Minimum methylation difference to call DMR
        qvalue: Q-value threshold for significance
        mc_cores: Number of cores for parallel processing
        verbose: Print progress messages
        
    Returns:
        dict with keys: 'tiles_all', 'tiles_significant', 'tile_stats'
    """
    import os
    import pandas as pd
    import rpy2.robjects as ro
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build R code strings
    files_str = ', '.join([f'"{f}"' for f in coverage_files])
    ids_str = ', '.join([f'"{s}"' for s in sample_ids])
    treat_str = ', '.join([str(t) for t in treatments])
    
    n_samples = len(sample_ids)
    
    if verbose:
        print("Step 1: Reading methylation data...")

    ro.r(f'''
        library(methylKit)
        
        file_list <- as.list(c({files_str}))
        sample_ids <- as.list(c({ids_str}))
        treatments <- c({treat_str})
        
        meth_obj <- methRead(
            file_list,
            sample.id = sample_ids,
            treatment = treatments,
            assembly = "hg38",
            pipeline = "bismarkCoverage",
            context = "CpG", 
            mincov = 1L,
            header = FALSE
        )
    ''')
    if verbose:
        print("  Methylation data loaded.")
        print("\n  Sample data preview for first sample:")
        ro.r('''
        # Check raw data from first sample
        dat <- getData(meth_obj[[1]])
        print(head(dat, 20))
        print(paste("Total rows:", nrow(dat)))
        ''')


    if verbose:
        print("Step 2: No coverage filtering...")
    
    # Diagnostic: CpGs per sample
    if verbose:
        print("\n  CpGs per sample with no filtering:")
        ro.r(f'''
            for (i in 1:{n_samples}) {{
                cat(paste0("    ", sample_ids[[i]], ": ", nrow(meth_obj[[i]]), " CpGs\\n"))
            }}
        ''')

    if verbose:
        print(f"\nStep 3: Aggregating into {tile_window}bp windows...")
    
    ro.r(f'''
        # Subset to only chr21 before tiling
        for (i in 1:length(meth_obj)) {{
            dat <- getData(meth_obj[[i]])
            chr21_idx <- which(dat$chr == "chr21")
            meth_obj[[i]] <- meth_obj[[i]][chr21_idx, ]
        }}
         
        # Tile/window the methylation counts to aggregate sparse CpGs
        tiles <- tileMethylCounts(
            meth_obj, 
            win.size = {tile_window}, 
            step.size = {tile_step},
            cov.bases = 1
        )
    ''')
    
    if verbose:
        print("Step 4: Filtering tiles by coverage...")
    
    ro.r(f'''
        tiles_filtered <- filterByCoverage(
            tiles,
            lo.count = {tile_min_coverage},
            lo.perc = NULL,
            hi.count = NULL,
            hi.perc = NULL
        )
    ''')
    
    # Diagnostic: tiles per sample
    if verbose:
        print("\n  Tiles per sample after filtering:")
        ro.r(f'''
            for (i in 1:{n_samples}) {{
                cat(paste0("    ", sample_ids[[i]], ": ", nrow(tiles_filtered[[i]]), " tiles\\n"))
            }}
        ''')
    

    if verbose:
        print("Step 4b: Exporting tiled methylation matrix...")

    tile_matrix_path = os.path.join(output_dir, "tile_methylation_matrix.csv")

    ro.r(f'''
        # Unite all samples (keep all tiles, even with missing data)
        tiles_for_export <- unite(tiles_filtered, destrand = FALSE, min.per.group = 0L)
        
        # Extract methylation percentages per sample
        tile_data <- getData(tiles_for_export)
        
        # Build methylation rate matrix
        # Columns: chr, start, end, then one column per sample with meth rate
        n_samples <- {n_samples}
        
        # The unite object has columns: chr, start, end, strand, then for each sample:
        # coverage1, numCs1, numTs1, coverage2, numCs2, numTs2, etc.
        
        meth_matrix <- data.frame(
            tile_id = paste(tile_data$chr, tile_data$start, tile_data$end, sep="_")
        )
        
        for (i in 1:n_samples) {{
            cov_col <- paste0("coverage", i)
            numCs_col <- paste0("numCs", i)
            
            if (cov_col %in% colnames(tile_data)) {{
                meth_rate <- tile_data[[numCs_col]] / tile_data[[cov_col]]
                meth_matrix[[sample_ids[[i]]]] <- meth_rate
            }}
        }}
        
        write.csv(meth_matrix, "{tile_matrix_path}", row.names = FALSE)
    ''')

    tile_matrix_df = pd.read_csv(tile_matrix_path)
    tile_matrix_df = tile_matrix_df.set_index('tile_id').T  # Transpose: samples as rows, tiles as columns

    if verbose:
        print(f"  Tile matrix shape: {tile_matrix_df.shape}")


    if verbose:
        print("\nStep 5: Normalizing coverage...")
    
    ro.r('''
        tiles_norm <- normalizeCoverage(tiles_filtered, method = "median")
    ''')
    

    if verbose:
        print("Step 5b: Exporting normalized methylation matrix...")

    norm_matrix_path = os.path.join(output_dir, "tile_methylation_normalized.csv")

    ro.r(f'''
        # Unite normalized tiles
        tiles_for_export <- unite(tiles_norm, destrand = FALSE, min.per.group = 0L)
        tile_data <- getData(tiles_for_export)
        
        # Build matrix with methylation rates from normalized data
        meth_matrix <- data.frame(
            tile_id = paste(tile_data$chr, tile_data$start, tile_data$end, sep="_")
        )
        
        for (i in 1:{n_samples}) {{
            cov_col <- paste0("coverage", i)
            numCs_col <- paste0("numCs", i)
            
            if (cov_col %in% colnames(tile_data)) {{
                meth_rate <- tile_data[[numCs_col]] / tile_data[[cov_col]]
                meth_matrix[[sample_ids[[i]]]] <- meth_rate
            }}
        }}
        
        write.csv(meth_matrix, "{norm_matrix_path}", row.names = FALSE)
    ''')

    norm_matrix_df = pd.read_csv(norm_matrix_path, index_col='tile_id').T


    if verbose:
        print("Step 6: Merging samples...")
    
    ro.r(f'''
        tiles_united <- unite(tiles_norm, destrand = FALSE, min.per.group = {min_per_group}L)
    ''')
    
    # Check if we have data
    n_tiles = ro.r('nrow(tiles_united)')[0]
    if verbose:
        print(f"\n  United tiles across samples: {int(n_tiles)}")
    
    if n_tiles == 0:
        print("\nWARNING: No tiles survived filtering. Try:")
        print("  - Decreasing tile_min_coverage")
        print("  - Increasing tile_window size")
        print("  - Decreasing min_per_group")
        return {
            'tiles_all': None,
            'tiles_significant': None,
            'tile_stats': {'n_tiles': 0}
        }
    
    if verbose:
        print("\nStep 7: Calculating differential methylation...")
    
    ro.r(f'''
        diff_tiles <- calculateDiffMeth(
            tiles_united,
            mc.cores = {mc_cores},
            test = "fast.fisher",
            adjust = "BH"
        )
    ''')
    
    if verbose:
        print("Step 8: Extracting results...")
    
    ro.r(f'''
        # All tiles with stats
        all_tiles <- getMethylDiff(diff_tiles, difference = 0, qvalue = 1)
        
        # Significant DMRs
        sig_tiles <- getMethylDiff(diff_tiles, difference = {min_diff * 100}, qvalue = {qvalue})
    ''')
    
    if verbose:
        print("Step 9: Converting results to pandas...")
    
    # Save to CSV in R, read back in Python (more reliable than rpy2 conversion)
    all_tiles_path = os.path.join(output_dir, "all_tiles.csv")
    sig_tiles_path = os.path.join(output_dir, "significant_dmrs.csv")
    
    ro.r(f'''
        write.csv(as.data.frame(all_tiles), "{all_tiles_path}", row.names=FALSE)
        write.csv(as.data.frame(sig_tiles), "{sig_tiles_path}", row.names=FALSE)
    ''')
    
    all_tiles_df = pd.read_csv(all_tiles_path)
    sig_tiles_df = pd.read_csv(sig_tiles_path)
    
    # Add direction column
    if len(sig_tiles_df) > 0:
        sig_tiles_df['direction'] = sig_tiles_df['meth.diff'].apply(
            lambda x: 'hyper' if x > 0 else 'hypo'
        )
        # Save updated version
        sig_tiles_df.to_csv(sig_tiles_path, index=False)
    
    # Summary statistics
    tile_stats = {
        'n_tiles_tested': len(all_tiles_df),
        'n_significant': len(sig_tiles_df),
        'n_hyper': len(sig_tiles_df[sig_tiles_df['meth.diff'] > 0]) if len(sig_tiles_df) > 0 else 0,
        'n_hypo': len(sig_tiles_df[sig_tiles_df['meth.diff'] < 0]) if len(sig_tiles_df) > 0 else 0,
        'window_size': tile_window,
        'min_coverage': tile_min_coverage
    }
    
    if verbose:
        print(f"\nResults summary:")
        print(f"  Window size: {tile_window}bp")
        print(f"  Total tiles tested: {tile_stats['n_tiles_tested']}")
        print(f"  Significant DMRs (|diff|>{min_diff*100}%, q<{qvalue}): {tile_stats['n_significant']}")
        print(f"    Hypermethylated in ALS: {tile_stats['n_hyper']}")
        print(f"    Hypomethylated in ALS: {tile_stats['n_hypo']}")
        print(f"\nResults saved to {output_dir}")
    
    return {
        'tiles_all': all_tiles_df,
        'tiles_significant': sig_tiles_df,
        'tile_stats': tile_stats,
        'tile_matrix': tile_matrix_df,
        'norm_tile_matrix': norm_matrix_df
    }



#################################
# IMPORT RESULTS BACK TO PANDAS #
#################################

def load_methylkit_results(output_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load methylKit results from output directory.
    
    Returns:
        Dictionary with DataFrames: 'all_positions', 'significant_dmps', 'dmrs'
    """
    results = {}
    
    all_pos_path = os.path.join(output_dir, "all_positions.csv")
    if os.path.exists(all_pos_path):
        results['all_positions'] = pd.read_csv(all_pos_path)
    
    sig_dmp_path = os.path.join(output_dir, "significant_dmps.csv")
    if os.path.exists(sig_dmp_path):
        results['significant_dmps'] = pd.read_csv(sig_dmp_path)
    
    dmr_path = os.path.join(output_dir, "dmrs.csv")
    if os.path.exists(dmr_path):
        results['dmrs'] = pd.read_csv(dmr_path)
    
    return results


def summarize_methylkit_results(results: Dict[str, pd.DataFrame]) -> str:
    """Generate summary of methylKit results."""
    lines = []
    lines.append("=" * 60)
    lines.append("METHYLKIT RESULTS SUMMARY")
    lines.append("=" * 60)
    
    if 'all_positions' in results:
        df = results['all_positions']
        lines.append(f"\nPositions tested: {len(df):,}")
        if 'qvalue' in df.columns:
            sig = (df['qvalue'] < 0.05).sum()
            lines.append(f"Significant (q < 0.05): {sig:,}")
    
    if 'significant_dmps' in results:
        df = results['significant_dmps']
        lines.append(f"\nSignificant DMPs: {len(df):,}")
        if 'meth.diff' in df.columns:
            hyper = (df['meth.diff'] > 0).sum()
            hypo = (df['meth.diff'] < 0).sum()
            lines.append(f"  Hypermethylated: {hyper}")
            lines.append(f"  Hypomethylated: {hypo}")
    
    if 'dmrs' in results:
        df = results['dmrs']
        lines.append(f"\nDMRs: {len(df)}")
        if len(df) > 0 and 'meth.diff' in df.columns:
            hyper = (df['meth.diff'] > 0).sum()
            hypo = (df['meth.diff'] < 0).sum()
            lines.append(f"  Hypermethylated: {hyper}")
            lines.append(f"  Hypomethylated: {hypo}")
            
            lines.append(f"\nTop DMRs by significance:")
            top = df.nsmallest(5, 'qvalue') if 'qvalue' in df.columns else df.head()
            for _, row in top.iterrows():
                lines.append(
                    f"  {row.get('chr', 'NA')}:{row.get('start', 'NA')}-{row.get('end', 'NA')} "
                    f"diff={row.get('meth.diff', 'NA'):.1f}%, q={row.get('qvalue', 'NA'):.2e}"
                )
    
    return "\n".join(lines)