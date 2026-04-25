"""
Part A: Oxidation Susceptibility Audit
======================================

For each molecular component in SHiP vs current mRNA-LNP formulations,
compute:
  (1) Weakest C-H bond dissociation energy (BDE) accessible to oxidation
  (2) Number of such "oxidation-initiation" sites per molecule
  (3) Relative oxidation initiation rate at 25 C via Arrhenius

Output: ranked comparison plot + CSV + summary statistics.

Method:
- BDE values are taken from peer-reviewed literature (Luo 2007;
  Yin/Xu/Porter 2011; Chem. Rev. 2024 membrane lipid chemistry).
- Number of bis-allylic / allylic sites counted from molecular structure.
- Relative rate is proportional to: n_sites * exp(-BDE / RT).
  This is a FIRST-ORDER KINETIC PROXY, not a full simulation; it captures
  the exponential sensitivity of initiation to BDE which is the dominant
  factor at ambient conditions.

Interpretation guidance:
- Rates are reported RELATIVE to an aliphatic C-H reference (BDE = 98 kcal/mol,
  1 site per molecule). Absolute rates depend on initiator concentration,
  oxygen partial pressure, etc. The RATIO between components is the stable,
  physically meaningful comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ================================================================
# Physical constants
# ================================================================
R_kcal = 1.987e-3      # Gas constant in kcal/(mol*K)
T_storage = 298.15     # 25 C in Kelvin

# ================================================================
# Component database
# ================================================================
# Each component: (name, weakest_BDE_kcal_mol, n_sites, category, notes)
# BDE values reference Luo (2007), Yin et al. Chem. Rev. 2011,
# and Galano/Alvarez-Idaboy review on lipid peroxidation.

# NOTE: "n_sites" counts positions where a hydrogen can be abstracted
# by a peroxyl radical at the given BDE or lower. For polymer components
# we count per repeat unit then scale by polymer length where relevant.

components = [
    # ========== Current LNP components ==========
    {
        "name": "SM-102",
        "category": "LNP (Moderna)",
        "bde": 75.0,
        "n_sites": 8,      # Multiple bis-allylic sites in branched unsaturated tails
        "notes": "Ionizable lipid. 2+ C=C bonds per tail; bis-allylic H between them.",
        "in_formulation": "SHiP_no_LNP_yes"
    },
    {
        "name": "ALC-0315",
        "category": "LNP (Pfizer)",
        "bde": 75.0,
        "n_sites": 6,      # Branched-chain with internal unsaturation
        "notes": "Ionizable lipid in BNT162b2. Bis-allylic sites in branched tails.",
        "in_formulation": "SHiP_no_LNP_yes"
    },
    {
        "name": "DLin-MC3-DMA",
        "category": "LNP (Patisiran)",
        "bde": 75.0,
        "n_sites": 8,      # Two linoleyl tails, each with bis-allylic site
        "notes": "Two linoleyl chains (9,12-diene each) → strongly bis-allylic.",
        "in_formulation": "SHiP_no_LNP_yes"
    },
    {
        "name": "Cholesterol",
        "category": "LNP helper",
        "bde": 83.0,
        "n_sites": 2,      # Allylic positions on sterol B-ring (C7)
        "notes": "Allylic C-H at C7 (adjacent to C5=C6 double bond). Oxidizes to 7-ketocholesterol, 7-OH-cholesterol.",
        "in_formulation": "SHiP_no_LNP_yes"
    },
    {
        "name": "DSPC",
        "category": "LNP helper",
        "bde": 98.0,
        "n_sites": 2,      # Fully saturated; only generic aliphatic C-H
        "notes": "Distearoylphosphatidylcholine - saturated phospholipid. Oxidation-resistant.",
        "in_formulation": "SHiP_no_LNP_yes"
    },
    # ========== SHiP components ==========
    {
        "name": "Lysine (in pHK)",
        "category": "SHiP polymer",
        "bde": 96.5,
        "n_sites": 1,      # Alpha-C-H of amino acid (slightly activated by N, C=O)
        "notes": "Alpha carbon of amino acid. BDE ~96-97 kcal/mol - similar to glycine. No bis-allylic sites.",
        "in_formulation": "SHiP_yes_LNP_no"
    },
    {
        "name": "Histidine (in pHK)",
        "category": "SHiP polymer",
        "bde": 94.0,
        "n_sites": 1,      # Alpha-C-H slightly activated by imidazole ring
        "notes": "Alpha-C-H slightly weaker than lysine due to imidazole delocalization. Still far above lipid BDEs.",
        "in_formulation": "SHiP_yes_LNP_no"
    },
    {
        "name": "Silica shell (SiO₂)",
        "category": "SHiP shell",
        "bde": 108.0,       # Si-O bond energy; Si already fully oxidized
        "n_sites": 0,      # No C-H bonds at all
        "notes": "Inorganic. Silicon already in +4 oxidation state. Thermodynamically oxidation-terminal.",
        "in_formulation": "SHiP_yes_LNP_no"
    },
    {
        "name": "PEG (per repeat unit)",
        "category": "SHiP surface",
        "bde": 93.0,        # Alpha-C-H adjacent to ether oxygen
        "n_sites": 2,       # Two CH2 per ethylene glycol unit
        "notes": "Alpha-C-H to ether is weakly activated. Much slower than bis-allylic. Also used in LNPs (not a differentiator).",
        "in_formulation": "both"
    },
    {
        "name": "Mannose",
        "category": "SHiP surface",
        "bde": 95.0,        # Anomeric / alpha-to-OH positions
        "n_sites": 1,       # Anomeric C-H weakest, ~95 kcal/mol
        "notes": "Sugar. Anomeric C-H is weakest. Slow relative to lipids.",
        "in_formulation": "SHiP_yes_LNP_no"
    },
    {
        "name": "Trehalose (matrix)",
        "category": "SHiP matrix",
        "bde": 95.0,        # Two anomeric C-H positions (both glucose)
        "n_sites": 2,       # One per glucose anomeric carbon
        "notes": "Non-reducing disaccharide. Two anomeric C-H. Oxidation-resistant enough to be used as antioxidant-mimetic.",
        "in_formulation": "SHiP_yes_LNP_no"
    },
    # ========== Shared / payload ==========
    {
        "name": "mRNA (backbone)",
        "category": "payload",
        "bde": 99.0,        # Ribose C-H bonds - unactivated
        "n_sites": 1,
        "notes": "RNA backbone C-H is ordinary aliphatic. Hydrolysis (not oxidation) is dominant RNA failure mode.",
        "in_formulation": "both"
    },
]

df = pd.DataFrame(components)

# ================================================================
# Calculate relative oxidation rate
# ================================================================
# Reference: pure aliphatic C-H at BDE = 98 kcal/mol, n_sites = 1
BDE_REF = 98.0
N_REF = 1

# k_rel = n_sites * exp(-(BDE - BDE_REF) / RT)
# Lower BDE -> larger positive number in exponent -> higher rate
df["delta_BDE"] = df["bde"] - BDE_REF
df["arrhenius_factor"] = np.exp(-df["delta_BDE"] / (R_kcal * T_storage))
# Handle the silica shell: 0 sites = 0 rate
df["relative_rate"] = df["n_sites"] * df["arrhenius_factor"]
# Cap silica at "effectively zero" for log plotting
df["relative_rate_plot"] = df["relative_rate"].replace(0, 1e-6)
df["log10_rel_rate"] = np.log10(df["relative_rate_plot"])

# Sort by relative rate, descending (most oxidizable first)
df_sorted = df.sort_values("relative_rate", ascending=False).reset_index(drop=True)

# ================================================================
# Print the table
# ================================================================
print("=" * 95)
print("PART A: OXIDATION SUSCEPTIBILITY AUDIT — SHiP vs current LNP components")
print("=" * 95)
print(f"{'Component':<28} {'Category':<20} {'BDE':>8} {'n_sites':>8} {'Rel rate':>14} {'log10':>8}")
print("-" * 95)
for _, row in df_sorted.iterrows():
    bde_str = f"{row['bde']:.1f}"
    rate_str = f"{row['relative_rate']:.2e}"
    log_str = f"{row['log10_rel_rate']:+.2f}"
    print(f"{row['name']:<28} {row['category']:<20} {bde_str:>8} {row['n_sites']:>8} {rate_str:>14} {log_str:>8}")
print("-" * 95)

# Save CSV
df_sorted.to_csv("/home/claude/ship_comp/oxidation_audit.csv", index=False)
print("\nSaved: oxidation_audit.csv")

# ================================================================
# Key summary statistics for the deck
# ================================================================
print("\n" + "=" * 95)
print("KEY STATISTICS FOR DECK / PROPOSAL")
print("=" * 95)

lnp_comps = df[df["in_formulation"].isin(["SHiP_no_LNP_yes", "both"])]
ship_comps = df[df["in_formulation"].isin(["SHiP_yes_LNP_no", "both"])]

max_lnp_rate = lnp_comps["relative_rate"].max()
max_ship_rate = ship_comps["relative_rate"].max()
ratio = max_lnp_rate / max_ship_rate if max_ship_rate > 0 else float('inf')

print(f"""
Most oxidation-susceptible component in LNP:    {lnp_comps.loc[lnp_comps['relative_rate'].idxmax(), 'name']}
  Relative initiation rate:                     {max_lnp_rate:.2e}

Most oxidation-susceptible component in SHiP:   {ship_comps.loc[ship_comps['relative_rate'].idxmax(), 'name']}
  Relative initiation rate:                     {max_ship_rate:.2e}

RATIO (LNP worst / SHiP worst):                 {ratio:.0f}x

Total bis-allylic sites (BDE < 80):
  LNP formulation:  {int(df[(df['bde'] < 80) & (df['in_formulation'].isin(['SHiP_no_LNP_yes', 'both']))]['n_sites'].sum())}
  SHiP formulation: {int(df[(df['bde'] < 80) & (df['in_formulation'].isin(['SHiP_yes_LNP_no', 'both']))]['n_sites'].sum())}
""")

# Save summary text too
with open("/home/claude/ship_comp/oxidation_summary.txt", "w") as f:
    f.write(f"Most oxidation-susceptible LNP component: {lnp_comps.loc[lnp_comps['relative_rate'].idxmax(), 'name']}\n")
    f.write(f"LNP max relative rate: {max_lnp_rate:.2e}\n")
    f.write(f"SHiP max relative rate: {max_ship_rate:.2e}\n")
    f.write(f"Ratio (LNP/SHiP): {ratio:.0f}x\n")

# ================================================================
# Visualization: log-scale horizontal bar chart
# ================================================================
fig, ax = plt.subplots(figsize=(12, 7.5), facecolor='#0B1929')
ax.set_facecolor('#0B1929')

# Color by formulation
color_map = {
    "SHiP_no_LNP_yes": "#EF4444",   # red - LNP only
    "SHiP_yes_LNP_no": "#06B6D4",   # cyan - SHiP only
    "both": "#94A3B8",               # silver - shared
}
colors = [color_map[row["in_formulation"]] for _, row in df_sorted.iterrows()]

# Take log10 of rate for plotting; silica (0) becomes -6
x_values = df_sorted["log10_rel_rate"].values
y_positions = np.arange(len(df_sorted))

bars = ax.barh(y_positions, x_values, color=colors, edgecolor='white', linewidth=0.5, alpha=0.95)

# Label each bar with the actual rate value
for i, (bar, rate) in enumerate(zip(bars, df_sorted["relative_rate"].values)):
    w = bar.get_width()
    # Place label on the outside of the bar: right side for positive, left side for negative
    if rate == 0:
        # Silica: bar extends to -6 (visual placeholder for "functionally zero").
        # Place label just to the right of x=0 so it doesn't overlap the y-axis label.
        label_x = 0.3
        ha = 'left'
        label = "≈ 0  (no C–H bonds at all)"
    else:
        label_x = w + (0.15 if w >= 0 else -0.15)
        ha = 'left' if w >= 0 else 'right'
        if rate >= 1e6:
            exp = int(np.floor(np.log10(rate)))
            mantissa = rate / (10 ** exp)
            label = f"{mantissa:.1f} × 10{str(exp).translate(str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻'))}"
        elif rate < 1e-3:
            label = f"{rate:.1e}"
        elif rate < 1:
            label = f"{rate:.3f}"
        elif rate < 100:
            label = f"{rate:.1f}"
        else:
            label = f"{rate:,.0f}"
    ax.text(label_x, bar.get_y() + bar.get_height()/2, label,
            va='center', ha=ha, color='white', fontsize=10, fontweight='bold')

# Y axis labels with category
ylabels = [f"{r['name']}\n({r['category']})" for _, r in df_sorted.iterrows()]
ax.set_yticks(y_positions)
ax.set_yticklabels(ylabels, color='white', fontsize=10)

ax.set_xlabel("log₁₀(relative oxidation initiation rate)\nreference: aliphatic C–H, BDE 98 kcal/mol, 1 site",
              color='white', fontsize=11, labelpad=12)

# Title
ax.set_title("Oxidation Susceptibility of SHiP vs mRNA-LNP Components\n(peroxyl-mediated H-abstraction rate, 25 °C)",
             color='white', fontsize=14, fontweight='bold', pad=20)

# Reference line at zero
ax.axvline(0, color='white', linestyle='--', alpha=0.3, linewidth=1)
ax.text(0.05, len(df_sorted) - 0.3, "reference\n(aliphatic C–H)",
        color='white', fontsize=8, alpha=0.6, style='italic')

# Spines and grid
for spine in ax.spines.values():
    spine.set_color('#334155')
ax.tick_params(colors='white')
ax.grid(axis='x', color='#1E293B', alpha=0.5, linewidth=0.5)
ax.invert_yaxis()  # most oxidizable at top

# Legend
legend_patches = [
    mpatches.Patch(color='#EF4444', label='In current LNPs only'),
    mpatches.Patch(color='#06B6D4', label='In SHiP only'),
    mpatches.Patch(color='#94A3B8', label='In both'),
]
ax.legend(handles=legend_patches, loc='lower right',
          facecolor='#152340', edgecolor='#334155',
          labelcolor='white', fontsize=10, framealpha=0.9)

# Annotation arrow showing the gap
# Find SM-102 and silica positions
ymax_lnp = df_sorted[df_sorted['name'] == 'DLin-MC3-DMA'].index[0]
ymin_ship = df_sorted[df_sorted['name'] == 'Silica shell (SiO₂)'].index[0]
top_rate = df_sorted.iloc[0]["relative_rate"]
top_log = df_sorted.iloc[0]["log10_rel_rate"]

# Big "10^13x gap" annotation
if ratio > 1e6:
    exponent = int(np.floor(np.log10(ratio)))
    # Unicode superscript translation
    sup_map = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')
    gap_text = f"~10{str(exponent).translate(sup_map)}× gap"
else:
    gap_text = f"{int(ratio):,}× gap"

ax.annotate(f"worst LNP vs worst\nSHiP component:\n{gap_text}",
            xy=(top_log + 0.3, ymax_lnp), xytext=(top_log - 5.5, 3),
            color='#FBBF24', fontsize=11, fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color='#FBBF24', lw=1.5, alpha=0.8),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#0B1929',
                      edgecolor='#FBBF24', linewidth=1.2))

plt.tight_layout()
plt.savefig("/home/claude/ship_comp/fig_oxidation_audit.png",
            dpi=200, facecolor='#0B1929', edgecolor='none',
            bbox_inches='tight', pad_inches=0.2)
plt.close()
print("Saved: fig_oxidation_audit.png")
