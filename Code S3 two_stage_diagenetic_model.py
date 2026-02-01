#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-Stage Diagenetic Model for Carbonate Trace Element Analysis
================================================================

This script implements a two-stage water-rock interaction model based on 
Banner & Hanson (1990) for interpreting trace element ratios in carbonate samples.

Reference:
    Banner, J.L., & Hanson, G.N. (1990). Calculation of simultaneous isotopic 
    and trace element variations during water-rock interaction with applications 
    to carbonate diagenesis. Geochimica et Cosmochimica Acta, 54(11), 3123-3137.

Features:
    - Early diagenesis pathway (W/R ratio: 0.01-10)
    - Late diagenesis pathway (W/R ratio: 0.5-200)
    - Intermediate transition pathways
    - W/R ratio isopleths
    - Ternary diagram visualization

Input:
    CSV file with columns: 'Mn/Sr', 'Mn/Fe', 'Sr/Ca'

Output:
    High-resolution ternary plot (300 DPI, publication-ready)

Usage:
    python two_stage_diagenetic_model.py <input.csv>
    
Author: [To be specified]
Date: 2026
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# ==============================================================================
# MODEL PARAMETERS
# ==============================================================================

# Early diagenesis stage parameters (shallow burial, oxic to suboxic conditions)
STAGE1_PARAMS = {
    'name': 'Early Diagenesis',
    'Cs0': {  # Initial solid phase concentrations (ppm)
        'Sr': 9874.0,
        'Mn': 0.735,
        'Fe': 164.0,
        'Ca': 380000.0
    },
    'Cf0': {  # Initial fluid phase concentrations (ppm)
        'Sr': 0.15,
        'Mn': 6.0,
        'Fe': 120.0
    },
    'D': {  # Partition coefficients (dimensionless)
        'Sr': 0.08,
        'Mn': 18.0,
        'Fe': 18.0
    },
    'N_range': [0.01, 10],  # Water/rock ratio range
    'color': 'crimson',
    'linewidth': 2.5,
    'linestyle': '-'
}

# Late diagenesis stage parameters (deep burial, reducing conditions)
STAGE2_PARAMS = {
    'name': 'Late Diagenesis',
    'Cs0': {  # Initial solid phase concentrations (ppm)
        'Sr': 5000.0,
        'Mn': 25.0,
        'Fe': 140.0,
        'Ca': 380000.0
    },
    'Cf0': {  # Initial fluid phase concentrations (ppm)
        'Sr': 0.05,
        'Mn': 15.0,
        'Fe': 200.0
    },
    'D': {  # Partition coefficients (dimensionless)
        'Sr': 0.04,
        'Mn': 25.0,
        'Fe': 22.0
    },
    'N_range': [0.5, 200],  # Water/rock ratio range
    'color': 'darkblue',
    'linewidth': 2.5,
    'linestyle': '-'
}

# W/R ratio values for isopleths
WR_ISOPLETHS = [0.5, 1, 2, 5, 10]

# Plotting parameters
PLOT_PARAMS = {
    'figsize': (10, 9),
    'dpi': 300,
    'data_color': 'royalblue',
    'data_size': 40,
    'data_alpha': 0.7,
    'intermediate_steps': 3,
    'pathway_resolution': 200
}

# ==============================================================================
# COMPUTATIONAL FUNCTIONS
# ==============================================================================

def calculate_concentrations(Cs0, Cf0, D, N_values):
    """
    Calculate solid phase trace element concentrations during diagenesis.
    
    Based on equation from Banner & Hanson (1990):
    Cs = (D * (Cs0 + N * Cf0)) / (D + N)
    
    Parameters:
    -----------
    Cs0 : dict
        Initial solid phase concentrations
    Cf0 : dict
        Initial fluid phase concentrations
    D : dict
        Partition coefficients
    N_values : array
        Water/rock ratio values
        
    Returns:
    --------
    DataFrame with Sr, Mn, Fe concentrations
    """
    results = {'Sr': [], 'Mn': [], 'Fe': []}
    
    for N in N_values:
        s_Sr = (D['Sr'] * (Cs0['Sr'] + N * Cf0['Sr'])) / (D['Sr'] + N)
        s_Mn = (D['Mn'] * (Cs0['Mn'] + N * Cf0['Mn'])) / (D['Mn'] + N)
        s_Fe = (D['Fe'] * (Cs0['Fe'] + N * Cf0['Fe'])) / (D['Fe'] + N)
        
        results['Sr'].append(s_Sr)
        results['Mn'].append(s_Mn)
        results['Fe'].append(s_Fe)
    
    return pd.DataFrame(results)


def get_ternary_coords(mn_sr, mn_fe, sr_ca_scaled):
    """
    Convert trace element ratios to Cartesian coordinates for ternary plot.
    
    Parameters:
    -----------
    mn_sr : array
        Mn/Sr ratios
    mn_fe : array
        Mn/Fe ratios
    sr_ca_scaled : array
        Sr/Ca ratios multiplied by 300
        
    Returns:
    --------
    x, y : arrays
        Cartesian coordinates
    """
    # Normalize components
    total = mn_sr + mn_fe + sr_ca_scaled
    norm_top = mn_sr / total
    norm_right = mn_fe / total
    
    # Convert barycentric to Cartesian coordinates
    x = 0.5 * norm_top + 1.0 * norm_right
    y = (np.sqrt(3) / 2) * norm_top
    
    return x, y


def generate_intermediate_params(stage1, stage2, num_steps=3):
    """
    Generate intermediate diagenetic pathways by linear interpolation.
    
    Parameters:
    -----------
    stage1 : dict
        Early diagenesis parameters
    stage2 : dict
        Late diagenesis parameters
    num_steps : int
        Number of intermediate pathways
        
    Returns:
    --------
    list of dicts containing intermediate pathway parameters
    """
    intermediates = []
    
    for i in range(1, num_steps + 1):
        fraction = i / (num_steps + 1)
        
        # Interpolate all parameters
        inter_Cs0 = {k: stage1['Cs0'][k] + fraction * (stage2['Cs0'][k] - stage1['Cs0'][k])
                     for k in stage1['Cs0']}
        inter_Cf0 = {k: stage1['Cf0'][k] + fraction * (stage2['Cf0'][k] - stage1['Cf0'][k])
                     for k in stage1['Cf0']}
        inter_D = {k: stage1['D'][k] + fraction * (stage2['D'][k] - stage1['D'][k])
                   for k in stage1['D']}
        
        n_min = stage1['N_range'][0] + fraction * (stage2['N_range'][0] - stage1['N_range'][0])
        n_max = stage1['N_range'][1] + fraction * (stage2['N_range'][1] - stage1['N_range'][1])
        
        intermediates.append({
            'name': f'Intermediate-{i}',
            'Cs0': inter_Cs0,
            'Cf0': inter_Cf0,
            'D': inter_D,
            'N_range': [n_min, n_max],
            'color': 'gray',
            'linewidth': 1.0,
            'linestyle': ':'
        })
    
    return intermediates


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_ternary_diagram(df, stage1, stage2, output_file='ternary_diagram.png'):
    """
    Generate publication-ready ternary diagram with diagenetic pathways.
    
    Parameters:
    -----------
    df : DataFrame
        Sample data with 'Mn/Sr', 'Mn/Fe', 'Sr/Ca' columns
    stage1 : dict
        Early diagenesis parameters
    stage2 : dict
        Late diagenesis parameters
    output_file : str
        Output filename
    """
    print("\n[INFO] Generating ternary diagram...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=PLOT_PARAMS['figsize'])
    
    # Draw ternary boundaries
    ax.plot([0, 1], [0, 0], 'k-', lw=1.5)
    ax.plot([0.5, 0], [np.sqrt(3)/2, 0], 'k-', lw=1.5)
    ax.plot([0.5, 1], [np.sqrt(3)/2, 0], 'k-', lw=1.5)
    
    # Generate pathways
    intermediates = generate_intermediate_params(
        stage1, stage2, 
        num_steps=PLOT_PARAMS['intermediate_steps']
    )
    all_pathways = [stage1] + intermediates + [stage2]
    
    # Store pathway data for isopleths
    pathway_data = []
    
    # Plot diagenetic pathways
    for stage in all_pathways:
        # Generate high-resolution W/R values
        N_values = np.logspace(
            np.log10(stage['N_range'][0]),
            np.log10(stage['N_range'][1]),
            PLOT_PARAMS['pathway_resolution']
        )
        
        # Calculate concentrations
        sim_df = calculate_concentrations(
            stage['Cs0'], stage['Cf0'], stage['D'], N_values
        )
        
        # Calculate ratios
        mn_sr = sim_df['Mn'] / sim_df['Sr']
        mn_fe = sim_df['Mn'] / sim_df['Fe']
        sr_ca_scaled = (sim_df['Sr'] / stage['Cs0']['Ca']) * 300
        
        # Convert to ternary coordinates
        x, y = get_ternary_coords(mn_sr.values, mn_fe.values, sr_ca_scaled.values)
        
        # Store for isopleths
        pathway_data.append({'N': N_values, 'x': x, 'y': y})
        
        # Plot pathway
        linestyle = stage.get('linestyle', '-')
        linewidth = stage.get('linewidth', 2.0)
        show_label = 'Intermediate' not in stage['name']
        
        ax.plot(x, y, 
                color=stage['color'], 
                lw=linewidth,
                linestyle=linestyle,
                label=stage['name'] if show_label else "")
        
        # Add directional arrow
        if show_label:
            ax.arrow(x[-2], y[-2], x[-1]-x[-2], y[-1]-y[-2],
                    shape='full', lw=0, length_includes_head=True,
                    head_width=0.02, color=stage['color'])
    
    # Plot W/R isopleths
    for wr in WR_ISOPLETHS:
        iso_x, iso_y = [], []
        
        for p_data in pathway_data:
            if p_data['N'].min() <= wr <= p_data['N'].max():
                idx = np.abs(p_data['N'] - wr).argmin()
                iso_x.append(p_data['x'][idx])
                iso_y.append(p_data['y'][idx])
        
        if len(iso_x) > 1:
            ax.plot(iso_x, iso_y, 'k--', lw=0.8, alpha=0.5)
            
            # Label isopleth
            if len(iso_x) == len(all_pathways):
                ax.text(iso_x[0]-0.02, iso_y[0]+0.005, 
                       f'W/R={wr}',
                       fontsize=8, color='dimgray')
    
    # Plot sample data
    data_mn_sr = df['Mn/Sr'].values
    data_mn_fe = df['Mn/Fe'].values
    data_sr_ca_scaled = df['Sr/Ca'].values * 300
    
    user_x, user_y = get_ternary_coords(data_mn_sr, data_mn_fe, data_sr_ca_scaled)
    
    ax.scatter(user_x, user_y, 
              c=PLOT_PARAMS['data_color'], 
              s=PLOT_PARAMS['data_size'],
              alpha=PLOT_PARAMS['data_alpha'],
              edgecolors='white', 
              linewidth=0.5,
              label='Sample Data',
              zorder=10)
    
    # Add axis labels
    ax.text(0.5, np.sqrt(3)/2 + 0.02, 'Mn/Sr', 
           ha='center', fontsize=12, fontweight='bold')
    ax.text(-0.02, -0.02, '300×Sr/Ca', 
           ha='right', fontsize=12, fontweight='bold')
    ax.text(1.02, -0.02, 'Mn/Fe', 
           ha='left', fontsize=12, fontweight='bold')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='crimson', lw=2.5, label='Early Diagenesis'),
        Line2D([0], [0], color='darkblue', lw=2.5, label='Late Diagenesis'),
        Line2D([0], [0], color='gray', lw=1, ls=':', label='Intermediate Pathways'),
        Line2D([0], [0], color='black', lw=0.8, ls='--', label='W/R Isopleths'),
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor='royalblue', markersize=8, label='Sample Data')
    ]
    
    ax.legend(handles=legend_elements, 
             loc='upper right', 
             frameon=True,
             fontsize=10,
             fancybox=True,
             shadow=True)
    
    # Set title
    ax.set_title('Two-Stage Diagenetic Model',
                fontsize=14, fontweight='bold', pad=20)
    
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Save figure
    plt.savefig(output_file, 
               dpi=PLOT_PARAMS['dpi'],
               bbox_inches='tight',
               facecolor='white',
               edgecolor='none')
    
    print(f"[INFO] Plot saved: {output_file}")
    plt.close()


def print_parameter_table():
    """Print formatted parameter table for publication."""
    
    print("\n" + "="*80)
    print("MODEL PARAMETERS")
    print("="*80)
    
    print("\n1. EARLY DIAGENESIS (Shallow burial, oxic to suboxic)")
    print("-" * 80)
    print(f"{'Parameter':<30} {'Symbol':<15} {'Value':<20} {'Unit':<15}")
    print("-" * 80)
    
    # Early diagenesis - Solid phase
    print("Initial Solid Phase Concentrations:")
    print(f"  {'Strontium':<28} {'Cs0(Sr)':<15} {STAGE1_PARAMS['Cs0']['Sr']:<20.2f} {'ppm':<15}")
    print(f"  {'Manganese':<28} {'Cs0(Mn)':<15} {STAGE1_PARAMS['Cs0']['Mn']:<20.3f} {'ppm':<15}")
    print(f"  {'Iron':<28} {'Cs0(Fe)':<15} {STAGE1_PARAMS['Cs0']['Fe']:<20.2f} {'ppm':<15}")
    print(f"  {'Calcium':<28} {'Cs0(Ca)':<15} {STAGE1_PARAMS['Cs0']['Ca']:<20.0f} {'ppm':<15}")
    
    # Early diagenesis - Fluid phase
    print("\nInitial Fluid Phase Concentrations:")
    print(f"  {'Strontium':<28} {'Cf0(Sr)':<15} {STAGE1_PARAMS['Cf0']['Sr']:<20.2f} {'ppm':<15}")
    print(f"  {'Manganese':<28} {'Cf0(Mn)':<15} {STAGE1_PARAMS['Cf0']['Mn']:<20.2f} {'ppm':<15}")
    print(f"  {'Iron':<28} {'Cf0(Fe)':<15} {STAGE1_PARAMS['Cf0']['Fe']:<20.2f} {'ppm':<15}")
    
    # Early diagenesis - Partition coefficients
    print("\nPartition Coefficients:")
    print(f"  {'Strontium':<28} {'D(Sr)':<15} {STAGE1_PARAMS['D']['Sr']:<20.2f} {'-':<15}")
    print(f"  {'Manganese':<28} {'D(Mn)':<15} {STAGE1_PARAMS['D']['Mn']:<20.2f} {'-':<15}")
    print(f"  {'Iron':<28} {'D(Fe)':<15} {STAGE1_PARAMS['D']['Fe']:<20.2f} {'-':<15}")
    
    print(f"\nWater/Rock Ratio Range: {STAGE1_PARAMS['N_range'][0]} - {STAGE1_PARAMS['N_range'][1]}")
    
    print("\n" + "="*80)
    print("\n2. LATE DIAGENESIS (Deep burial, reducing conditions)")
    print("-" * 80)
    print(f"{'Parameter':<30} {'Symbol':<15} {'Value':<20} {'Unit':<15}")
    print("-" * 80)
    
    # Late diagenesis - Solid phase
    print("Initial Solid Phase Concentrations:")
    print(f"  {'Strontium':<28} {'Cs0(Sr)':<15} {STAGE2_PARAMS['Cs0']['Sr']:<20.2f} {'ppm':<15}")
    print(f"  {'Manganese':<28} {'Cs0(Mn)':<15} {STAGE2_PARAMS['Cs0']['Mn']:<20.2f} {'ppm':<15}")
    print(f"  {'Iron':<28} {'Cs0(Fe)':<15} {STAGE2_PARAMS['Cs0']['Fe']:<20.2f} {'ppm':<15}")
    print(f"  {'Calcium':<28} {'Cs0(Ca)':<15} {STAGE2_PARAMS['Cs0']['Ca']:<20.0f} {'ppm':<15}")
    
    # Late diagenesis - Fluid phase
    print("\nInitial Fluid Phase Concentrations:")
    print(f"  {'Strontium':<28} {'Cf0(Sr)':<15} {STAGE2_PARAMS['Cf0']['Sr']:<20.2f} {'ppm':<15}")
    print(f"  {'Manganese':<28} {'Cf0(Mn)':<15} {STAGE2_PARAMS['Cf0']['Mn']:<20.2f} {'ppm':<15}")
    print(f"  {'Iron':<28} {'Cf0(Fe)':<15} {STAGE2_PARAMS['Cf0']['Fe']:<20.2f} {'ppm':<15}")
    
    # Late diagenesis - Partition coefficients
    print("\nPartition Coefficients:")
    print(f"  {'Strontium':<28} {'D(Sr)':<15} {STAGE2_PARAMS['D']['Sr']:<20.2f} {'-':<15}")
    print(f"  {'Manganese':<28} {'D(Mn)':<15} {STAGE2_PARAMS['D']['Mn']:<20.2f} {'-':<15}")
    print(f"  {'Iron':<28} {'D(Fe)':<15} {STAGE2_PARAMS['D']['Fe']:<20.2f} {'-':<15}")
    
    print(f"\nWater/Rock Ratio Range: {STAGE2_PARAMS['N_range'][0]} - {STAGE2_PARAMS['N_range'][1]}")
    print("="*80 + "\n")


def print_data_statistics(df):
    """Print sample data statistics."""
    
    print("\n" + "="*80)
    print("SAMPLE DATA STATISTICS")
    print("="*80)
    print(f"\nNumber of samples: {len(df)}")
    print("\nTrace Element Ratios:")
    print("-" * 80)
    print(f"{'Ratio':<15} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
    print("-" * 80)
    
    for col in ['Mn/Sr', 'Mn/Fe', 'Sr/Ca']:
        stats = df[col].describe()
        print(f"{col:<15} {stats['mean']:<12.4f} {stats['std']:<12.4f} "
              f"{stats['min']:<12.4f} {stats['max']:<12.4f}")
    
    print("="*80 + "\n")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main(csv_file, output_file='ternary_diagram.png'):
    """
    Main execution function.
    
    Parameters:
    -----------
    csv_file : str
        Path to input CSV file
    output_file : str
        Path to output figure file
    """
    print("\n" + "="*80)
    print("TWO-STAGE DIAGENETIC MODEL")
    print("="*80)
    print("\nReference: Banner & Hanson (1990), Geochimica et Cosmochimica Acta")
    
    try:
        # Load data
        print(f"\n[INFO] Loading data from: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Validate required columns
        required_cols = ['Mn/Sr', 'Mn/Fe', 'Sr/Ca']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"[INFO] Loaded {len(df)} samples successfully")
        
        # Print parameter table
        print_parameter_table()
        
        # Print data statistics
        print_data_statistics(df)
        
        # Generate plot
        plot_ternary_diagram(df, STAGE1_PARAMS, STAGE2_PARAMS, output_file)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nOutput file: {output_file}")
        print(f"Resolution: {PLOT_PARAMS['dpi']} DPI (publication-ready)\n")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python two_stage_diagenetic_model.py <input.csv> [output.png]")
        print("\nExample:")
        print("  python two_stage_diagenetic_model.py samples.csv diagram.png")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'ternary_diagram.png'
    
    main(csv_file, output_file)
