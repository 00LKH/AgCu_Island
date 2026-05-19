"""
Render presentation figures for the Ag/Ag(111) RSS + GCGA pipeline.

Outputs PNGs into ./figures/, used by methodology_slides.md.

Run:
    conda activate agox_uma
    python make_figures.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from ase.io import read
from ase.visualize.plot import plot_atoms

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

# Common ASE rendering helpers ------------------------------------------------

def render_pair(atoms, out_path, title=None, scale_top=1.0, scale_side=1.0):
    """Top view + side view side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=160)
    plot_atoms(atoms, axes[0], rotation="0x,0y,0z",
               show_unit_cell=2, radii=0.45 * scale_top)
    axes[0].set_title("Top view (z↓)")
    axes[0].set_axis_off()
    plot_atoms(atoms, axes[1], rotation="-90x,0y,0z",
               show_unit_cell=2, radii=0.45 * scale_side)
    axes[1].set_title("Side view")
    axes[1].set_axis_off()
    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out_path}")


def render_single(atoms, out_path, rotation="-75x,15y,0z", title=None, radii=0.6):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
    plot_atoms(atoms, ax, rotation=rotation,
               show_unit_cell=2, radii=radii)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out_path}")


# Stage 1: bulk + primitive slab + relaxed supercell --------------------------

print("[Stage 1] bulk + slab figures")
bulk = read(os.path.join(BASE, "00_slab", "1_Ag_bulk_relaxed.vasp"))
slab_prim = read(os.path.join(BASE, "00_slab", "2_Ag_slab.vasp"))
slab_super = read(os.path.join(BASE, "00_slab", "4_Ag_slab_supercell_opt.vasp"))

render_single(bulk * (3, 3, 3), os.path.join(FIG, "01_bulk.png"),
              rotation="-70x,20y,0z", title="Ag bulk (FCC, vc-relaxed)", radii=0.7)
render_pair(slab_prim, os.path.join(FIG, "02_slab_primitive.png"),
            title="Primitive Ag(111) slab")
render_pair(slab_super, os.path.join(FIG, "03_slab_supercell.png"),
            title="Ag(111) 16x16 supercell — 1280 atoms",
            scale_top=1.0 / 0.45, scale_side=1.0 / 0.45)  # radii = 1.0


# Stage 2: representative Ag32 cluster ----------------------------------------

print("[Stage 2] cluster figure")
cluster_dir = os.path.join(BASE, "01_cluster", "Ag_32_relaxed")
cluster_files = sorted([f for f in os.listdir(cluster_dir) if f.endswith(".vasp")])
# pick first relaxed cluster as a representative motif
cl = read(os.path.join(cluster_dir, cluster_files[0]))
render_single(cl, os.path.join(FIG, "04_cluster_Ag32.png"),
              rotation="-75x,15y,0z",
              title=f"Relaxed Ag32 cluster (e.g. {cluster_files[0]})", radii=0.85)


# Stage 3: one slab+cluster parent --------------------------------------------

print("[Stage 3] slab+cluster figure")
sc_dir = os.path.join(BASE, "02_slab_cluster", "generated_surfaces_relaxed")
sc_files = sorted([f for f in os.listdir(sc_dir) if f.endswith(".vasp")])
sc = read(os.path.join(sc_dir, sc_files[0]))
render_pair(sc, os.path.join(FIG, "05_slab_plus_clusters.png"),
            title=f"Relaxed parent: 8 x Ag32 on slab — {len(sc)} atoms",
            scale_top=1.0 / 0.45, scale_side=1.0 / 0.45)  # radii = 1.0


# Stage 4: lowest-energy GCGA structure ---------------------------------------

print("[Stage 4] GCGA minimum figure")
candidates = read(os.path.join(BASE, "03_gcga", "gcga.db"), index=":")
energies = np.array([a.get_potential_energy() for a in candidates])
idx_min = int(np.argmin(energies))
ga_min = candidates[idx_min]
render_pair(ga_min, os.path.join(FIG, "06_gcga_minimum.png"),
            title=f"GCGA lowest-E: idx={idx_min}, E={energies[idx_min]:.2f} eV, N={len(ga_min)}",
            scale_top=1.0 / 0.45, scale_side=1.0 / 0.45)  # radii = 1.0


# GCGA energy convergence plot ------------------------------------------------

print("[Stage 4] energy convergence plot")
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
order = np.arange(1, len(energies) + 1)
ax.plot(order, energies, ".", ms=4, color="#1f3a5f", alpha=0.6, label="candidates")
running_min = np.minimum.accumulate(energies)
ax.plot(order, running_min, "-", lw=2, color="#c0392b", label="running minimum")
ax.set_xlabel("structure index (db order)")
ax.set_ylabel("potential energy (eV)")
ax.set_title("GCGA energy trace (gcga.db)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "07_energy_trace.png"),
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  -> {os.path.join(FIG, '07_energy_trace.png')}")


# Pipeline schematic ----------------------------------------------------------

print("[schematic] pipeline overview")
fig, ax = plt.subplots(figsize=(12, 4), dpi=160)
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.set_axis_off()

stages = [
    ("00_slab",        "Slab prep",
     "Ag bulk → (111) slab\n16x16 supercell\nUMA-s-1p1 relax", "#2c3e50"),
    ("01_cluster",     "Cluster RSS",
     "AGOX RandomGenerator\nAg32 in vacuum box\nrelax → mu_Ag", "#2980b9"),
    ("02_slab_cluster","Random assembly",
     "8 x Ag32 random tiling\nPBC non-overlap (MIC)\n50 parent confs", "#27ae60"),
    ("03_gcga",        "GCGA evolution",
     "GOCIA grand-canonical GA\ncrossover/mutation + relax\nnatural selection", "#c0392b"),
]
w, h = 2.6, 2.4
y0 = 0.8
for i, (tag, title, body, color) in enumerate(stages):
    x0 = 0.15 + i * (w + 0.5)
    box = FancyBboxPatch((x0, y0), w, h,
                         boxstyle="round,pad=0.03,rounding_size=0.15",
                         linewidth=1.4, edgecolor=color, facecolor="white")
    ax.add_patch(box)
    ax.text(x0 + w / 2, y0 + h - 0.25, tag,
            ha="center", va="center", fontsize=9,
            color="white",
            bbox=dict(boxstyle="round,pad=0.18", fc=color, ec="none"))
    ax.text(x0 + w / 2, y0 + h - 0.75, title,
            ha="center", va="center", fontsize=13, fontweight="bold", color=color)
    ax.text(x0 + w / 2, y0 + 0.7, body,
            ha="center", va="center", fontsize=9, color="#222")

    if i < len(stages) - 1:
        x_arrow_start = x0 + w + 0.03
        x_arrow_end = x0 + w + 0.47
        arrow = FancyArrowPatch((x_arrow_start, y0 + h / 2),
                                (x_arrow_end, y0 + h / 2),
                                arrowstyle="-|>", mutation_scale=18,
                                linewidth=2, color="#555")
        ax.add_patch(arrow)

ax.text(6.0, 0.3,
        "Energy model throughout: FAIRChem UMA-s-1p1 (task='omat'),  seed = 42",
        ha="center", va="center", fontsize=10, style="italic", color="#444")

fig.savefig(os.path.join(FIG, "00_pipeline.png"),
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  -> {os.path.join(FIG, '00_pipeline.png')}")

print("\nDone. Figures in:", FIG)
