"""
plotting.py

plotting functions for methylation sample-level statistics

plots:
- motif proportions
- fragment length distribution

Requirements:
    pip install numpy matplotlib pyparsing

Usage:
    from plotting import plot_sample_distributions

    plot_sample_distributions(features, sample_name="Sample1")
"""



import matplotlib.pyplot as plt
import numpy as np

def plot_sample_distributions(features: Dict, sample_name: str = "Sample"):
    """Plot motif proportions and fragment length distribution for a sample
    
    Args: 
        features (Dict): Dictionary containing 'motif_4mer' and '_raw' fragment lengths. From aggregate_features_from_bam()
        sample_name (str): Name of the sample for titles
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Plot 1: Motif proportions ---
    ax1 = axes[0]
    motif_data = features.get('motif_4mer', {})
    
    if motif_data:
        # Sort by frequency, show top 20
        sorted_motifs = sorted(motif_data.items(), key=lambda x: x[1], reverse=True)
        top_n = 20
        top_motifs = sorted_motifs[:top_n]
        
        motifs = [m[0] for m in top_motifs]
        freqs = [m[1] for m in top_motifs]
        
        bars = ax1.bar(range(len(motifs)), freqs, color='steelblue')
        ax1.set_xticks(range(len(motifs)))
        ax1.set_xticklabels(motifs, rotation=45, ha='right', fontsize=9)
        ax1.set_xlabel('4-mer Motif')
        ax1.set_ylabel('Proportion')
        ax1.set_title(f'Top {top_n} 4-mer Motifs - {sample_name}')
    else:
        ax1.text(0.5, 0.5, 'No motif data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f'4-mer Motifs - {sample_name}')
    
    # --- Plot 2: Fragment length distribution ---
    ax2 = axes[1]
    lengths = features.get('_raw', {}).get('fragment_lengths', [])
    
    if lengths:
        ax2.hist(lengths, bins=50, color='coral', edgecolor='black', alpha=0.7)
        ax2.axvline(np.median(lengths), color='red', linestyle='--', label=f'Median: {np.median(lengths):.0f}')
        ax2.axvline(np.mean(lengths), color='blue', linestyle='--', label=f'Mean: {np.mean(lengths):.0f}')
        ax2.set_xlabel('Fragment Length (bp)')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Fragment Length Distribution - {sample_name}')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No fragment length data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'Fragment Length - {sample_name}')

    return fig, axes




import io
from IPython.display import display
from PIL import Image as PILImage

def combine_plots(plots, cols=1):
    """Combine multiple (fig, axes) tuples into one image.
    
    Args:
        plots (List[Tuple[Figure, Axes]]): List of (fig, axes) tuples to combine
        cols (int): Number of columns in the combined image
    """
    images = []
    
    for fig, axes in plots:
        # Force render all elements
        fig.canvas.draw()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, 
                    bbox_inches='tight',
                    pad_inches=0.3,  # Add padding to prevent clipping
                    facecolor='white',
                    edgecolor='none')
        buf.seek(0)
        images.append(PILImage.open(buf).copy())  # .copy() to keep image after buffer closes
        buf.close()
    
    rows = (len(images) + cols - 1) // cols
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    
    combined = PILImage.new('RGB', (max_width * cols, max_height * rows), 'white')
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        # Center each image in its cell
        x_offset = col * max_width + (max_width - img.width) // 2
        y_offset = row * max_height + (max_height - img.height) // 2
        combined.paste(img, (x_offset, y_offset))
    
    return combined



import plotly.graph_objects as go
import numpy as np

def plot_dmr_manhattan_interactive(dmr_df, qvalue_col='qvalue', diff_col='meth.diff',
                                    sig_threshold=0.05, title='DMR Manhattan Plot'):
    """
    Interactive Manhattan-style plot for DMR results using Plotly.
    
    Args:
        dmr_df: DataFrame with chr, start, end, qvalue, meth.diff columns
        qvalue_col: Column name for q-values
        diff_col: Column name for methylation difference
        sig_threshold: Q-value threshold for significance
        title: Plot title
    
    Returns:
        Plotly figure
    """
    dmr_df = dmr_df.copy()
    
    # Calculate position (midpoint of tile)
    dmr_df['pos'] = (dmr_df['start'] + dmr_df['end']) / 2
    dmr_df['pos_mb'] = dmr_df['pos'] / 1e6
    
    # Determine significance and color
    def get_color(row):
        if row[qvalue_col] >= sig_threshold:
            return 'gray'
        elif row[diff_col] > 0:
            return 'tomato'
        else:
            return 'steelblue'
    
    dmr_df['color'] = dmr_df.apply(get_color, axis=1)
    dmr_df['significant'] = dmr_df[qvalue_col] < sig_threshold
    
    # Marker size — larger for significant
    dmr_df['marker_size'] = dmr_df['significant'].apply(lambda x: 14 if x else 8)
    
    # Create hover text
    dmr_df['hover'] = dmr_df.apply(
        lambda r: (
            f"<b>Position:</b> {r['chr']}:{r['start']:,}-{r['end']:,}<br>"
            f"<b>Meth Diff:</b> {r[diff_col]:.2f}%<br>"
            f"<b>q-value:</b> {r[qvalue_col]:.2e}<br>"
            f"<b>p-value:</b> {r['pvalue']:.2e}<br>"
            f"<b>Significant:</b> {'Yes' if r['significant'] else 'No'}"
        ), axis=1
    )
    
    # Create figure
    fig = go.Figure()
    
    # Plot non-significant points first (behind)
    nonsig = dmr_df[~dmr_df['significant']]
    if len(nonsig) > 0:
        fig.add_trace(go.Scatter(
            x=nonsig['pos_mb'],
            y=nonsig[diff_col],
            mode='markers',
            name='Not significant',
            marker=dict(
                size=nonsig['marker_size'],
                color='gray',
                line=dict(width=0.5, color='darkgray'),
                opacity=0.5
            ),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=nonsig['hover']
        ))
    
    # Plot significant hypomethylated
    sig_hypo = dmr_df[(dmr_df['significant']) & (dmr_df[diff_col] < 0)]
    if len(sig_hypo) > 0:
        fig.add_trace(go.Scatter(
            x=sig_hypo['pos_mb'],
            y=sig_hypo[diff_col],
            mode='markers',
            name='Hypomethylated (ALS)',
            marker=dict(
                size=sig_hypo['marker_size'],
                color='steelblue',
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=sig_hypo['hover']
        ))
    
    # Plot significant hypermethylated
    sig_hyper = dmr_df[(dmr_df['significant']) & (dmr_df[diff_col] > 0)]
    if len(sig_hyper) > 0:
        fig.add_trace(go.Scatter(
            x=sig_hyper['pos_mb'],
            y=sig_hyper[diff_col],
            mode='markers',
            name='Hypermethylated (ALS)',
            marker=dict(
                size=sig_hyper['marker_size'],
                color='tomato',
                line=dict(width=1, color='black'),
                opacity=0.8
            ),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=sig_hyper['hover']
        ))
    
    # Add horizontal lines
    fig.add_hline(y=0, line_dash='solid', line_color='black', line_width=0.5)
    fig.add_hline(y=10, line_dash='dash', line_color='gray', opacity=0.5)
    fig.add_hline(y=-10, line_dash='dash', line_color='gray', opacity=0.5)
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Position on chr21 (Mb)',
        yaxis_title='Methylation Difference (%)<br>(ALS - Control)',
        hovermode='closest',
        template='plotly_white',
        width=1000,
        height=500,
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99
        )
    )
    
    return fig