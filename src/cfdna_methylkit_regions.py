

def run_methylkit_region_dmr_analysis(
    coverage_files: list,
    sample_ids: list,
    treatments: list,
    regions_bed: str,
    output_dir: str = "./methylkit_results",
    min_per_group: int = 1,
    region_min_coverage: int = 5,
    min_diff: float = 0.1,
    qvalue: float = 0.05,
    mc_cores: int = 1,
    verbose: bool = True
):
    """
    Run methylKit DMR analysis using annotated regions (e.g., CpG islands).
    
    Args:
        coverage_files: List of paths to Bismark coverage files
        sample_ids: List of sample IDs
        treatments: List of treatment values (1 for ALS, 0 for Control)
        regions_bed: Path to BED file with regions (chr, start, end, name)
        output_dir: Directory for output files
        min_per_group: Minimum samples per group with coverage
        region_min_coverage: Minimum coverage per region
        min_diff: Minimum methylation difference for significance
        qvalue: Q-value threshold
        mc_cores: Number of cores
        verbose: Print progress
    
    Returns:
        dict with region methylation matrix and DMR results
    """
    import os
    import pandas as pd
    import rpy2.robjects as ro
    
    os.makedirs(output_dir, exist_ok=True)
    
    files_str = ', '.join([f'"{f}"' for f in coverage_files])
    ids_str = ', '.join([f'"{s}"' for s in sample_ids])
    treat_str = ', '.join([str(t) for t in treatments])
    
    n_samples = len(sample_ids)
    
    if verbose:
        print("Step 1: Reading methylation data...")
    
    ro.r(f'''
        library(methylKit)
        library(GenomicRanges)
        
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
            mincov = 1
        )
    ''')
    # Diagnostic: CpGs per sample
    
    if verbose:
        print("\n  CpGs per sample with no filtering:")
        ro.r(f'''
            for (i in 1:{n_samples}) {{
                cat(paste0("    ", sample_ids[[i]], ": ", nrow(meth_obj[[i]]), " CpGs\\n"))
            }}
        ''')

    
    if verbose:
        print(f"Step 2: Loading regions from {regions_bed}...")
    
    ro.r(f'''
        # Read BED file
        regions <- read.table("{regions_bed}", header=FALSE, sep="\\t",
                              stringsAsFactors=FALSE)
        
        # Handle 3-column or 4-column BED
        if (ncol(regions) >= 4) {{
            colnames(regions)[1:4] <- c("chr", "start", "end", "name")
        }} else {{
            colnames(regions)[1:3] <- c("chr", "start", "end")
            regions$name <- paste0(regions$chr, "_", regions$start, "_", regions$end)
        }}
        
        # Convert to GRanges (BED is 0-based, GRanges is 1-based)
        regions_gr <- GRanges(
            seqnames = regions$chr,
            ranges = IRanges(start = regions$start + 1, end = regions$end),
            name = regions$name
        )
        
        cat(paste0("  Loaded ", length(regions_gr), " regions\\n"))
    ''')
    
    if verbose:
        print("Step 3: Aggregating methylation by regions...")
    
    ro.r('''
        # Aggregate CpG methylation into regions
        region_meth <- regionCounts(meth_obj, regions_gr, cov.bases=1)
    ''')
    
    if verbose:
        print("Step 4: Filtering regions by coverage...")
    
    ro.r(f'''
        region_filtered <- filterByCoverage(
            region_meth,
            lo.count = {region_min_coverage},
            lo.perc = NULL,
            hi.count = NULL,
            hi.perc = NULL
        )
    ''')
    
    if verbose:
        print("\n  Regions per sample after filtering:")
        ro.r(f'''
            for (i in 1:{n_samples}) {{
                cat(paste0("    ", sample_ids[[i]], ": ", nrow(region_filtered[[i]]), " regions\\n"))
            }}
        ''')
    
    if verbose:
        print("\nStep 5: Normalizing coverage...")
    
    ro.r('''
        region_norm <- normalizeCoverage(region_filtered, method = "median")
    ''')
    
    if verbose:
        print("Step 6: Merging samples...")
    
    ro.r(f'''
        region_united <- unite(region_norm, destrand = FALSE, min.per.group = {min_per_group}L)
    ''')
    
    n_regions = ro.r('nrow(region_united)')[0]
    if verbose:
        print(f"\n  United regions across samples: {int(n_regions)}")
    
    if n_regions == 0:
        print("\nWARNING: No regions survived filtering.")
        return {
            'regions_all': None,
            'regions_significant': None,
            'region_matrix': None,
            'region_stats': {'n_regions': 0}
        }
    
    if verbose:
        print("\nStep 7: Calculating differential methylation...")
    
    ro.r(f'''
        diff_regions <- calculateDiffMeth(
            region_united,
            mc.cores = {mc_cores},
            test = "Chisq",
            adjust = "BH"
        )
    ''')
    
    if verbose:
        print("Step 8: Extracting results...")
    
    ro.r(f'''
        all_regions <- getMethylDiff(diff_regions, difference = 0, qvalue = 1)
        sig_regions <- getMethylDiff(diff_regions, difference = {min_diff * 100}, qvalue = {qvalue})
    ''')
    
    if verbose:
        print("Step 9: Exporting methylation matrix...")
    
    matrix_path = os.path.join(output_dir, "region_methylation_matrix.csv")
    
    ro.r(f'''
        region_data <- getData(region_united)
        
        # Build methylation rate matrix
        meth_matrix <- data.frame(
            region_id = paste(region_data$chr, region_data$start, region_data$end, sep="_")
        )
        
        for (i in 1:{n_samples}) {{
            cov_col <- paste0("coverage", i)
            numCs_col <- paste0("numCs", i)
            
            if (cov_col %in% colnames(region_data)) {{
                meth_rate <- region_data[[numCs_col]] / region_data[[cov_col]]
                meth_matrix[[sample_ids[[i]]]] <- meth_rate
            }}
        }}
        
        write.csv(meth_matrix, "{matrix_path}", row.names = FALSE)
    ''')
    
    # Save DMR results
    all_path = os.path.join(output_dir, "all_regions.csv")
    sig_path = os.path.join(output_dir, "significant_regions.csv")
    
    ro.r(f'''
        write.csv(as.data.frame(all_regions), "{all_path}", row.names=FALSE)
        write.csv(as.data.frame(sig_regions), "{sig_path}", row.names=FALSE)
    ''')
    
    # Read results back into Python
    all_regions_df = pd.read_csv(all_path)
    sig_regions_df = pd.read_csv(sig_path)
    region_matrix_df = pd.read_csv(matrix_path, index_col='region_id').T
    
    # Add direction column
    if len(sig_regions_df) > 0:
        sig_regions_df['direction'] = sig_regions_df['meth.diff'].apply(
            lambda x: 'hyper' if x > 0 else 'hypo'
        )
        sig_regions_df.to_csv(sig_path, index=False)
    
    region_stats = {
        'n_regions_tested': len(all_regions_df),
        'n_significant': len(sig_regions_df),
        'n_hyper': len(sig_regions_df[sig_regions_df['meth.diff'] > 0]) if len(sig_regions_df) > 0 else 0,
        'n_hypo': len(sig_regions_df[sig_regions_df['meth.diff'] < 0]) if len(sig_regions_df) > 0 else 0,
    }
    
    if verbose:
        print(f"\nResults summary:")
        print(f"  Total regions tested: {region_stats['n_regions_tested']}")
        print(f"  Significant regions: {region_stats['n_significant']}")
        print(f"    Hypermethylated: {region_stats['n_hyper']}")
        print(f"    Hypomethylated: {region_stats['n_hypo']}")
        print(f"\nResults saved to {output_dir}")
    
    return {
        'regions_all': all_regions_df,
        'regions_significant': sig_regions_df,
        'region_matrix': region_matrix_df,
        'region_stats': region_stats
    }