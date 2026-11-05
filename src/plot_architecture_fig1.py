"""Generator diagram arsitektur vektor Fig. 1 (IEEE Double-Column 7.16 in, 300 DPI PNG + PDF)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def _add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    facecolor: str = "#ffffff",
    edgecolor: str = "#2c3e50",
    linewidth: float = 0.9,
    fontsize: float = 7.4,
    pad: float = 0.45,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size=0.8",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1a252f",
        linespacing=1.28,
        zorder=4,
    )


def _add_arrow(
    ax: plt.Axes,
    p1: tuple[float, float],
    p2: tuple[float, float],
    color: str = "#34495e",
    lw: float = 1.05,
    style: str = "-|>",
    mutation_scale: float = 9.5,
    linestyle: str = "-",
) -> None:
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=linestyle,
        color=color,
        shrinkA=2.0,
        shrinkB=2.0,
        zorder=5,
    )
    ax.add_patch(arr)


def generate_fig1_architecture(figures_dir: Path = DEFAULT_FIGURES_DIR) -> tuple[Path, Path]:
    """Bangkitkan Fig. 1 Dual-Input Asymmetric SPECTER2 Retrieval Pipeline (7.16 x 3.00 in, 300 DPI)."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
    })

    # Kanvas 2 kolom IEEE presisi: lebar 7.16 in, rasio koordinat 100 x 42 proporsional tanpa distorsi
    fig, ax = plt.subplots(figsize=(7.16, 3.00), dpi=300)
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 42.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # === 1. Container Atas: Offline Document Indexing Pipeline ===
    offline_bg = FancyBboxPatch(
        (1.0, 25.0),
        98.0,
        15.8,
        boxstyle="round,pad=0.3,rounding_size=1.0",
        facecolor="#f4f7fb",
        edgecolor="#9fb3c8",
        linewidth=0.85,
        linestyle="--",
        zorder=1,
    )
    ax.add_patch(offline_bg)
    ax.text(
        2.4,
        39.2,
        "OFFLINE DOCUMENT INDEXING PIPELINE  (Executed Once on GPU, FP16 Dynamic Batching)",
        ha="left",
        va="center",
        fontsize=7.3,
        fontweight="bold",
        color="#1f4e78",
        zorder=2,
    )

    _add_box(
        ax, 2.5, 26.6, 18.0, 10.2,
        "LitSearch Corpus\n(64,183 CS/AI Papers)\nTitles & Abstracts",
        facecolor="#ffffff", edgecolor="#5d6d7e", fontsize=7.2,
    )
    _add_arrow(ax, (21.0, 31.7), (24.2, 31.7))

    _add_box(
        ax, 24.7, 26.6, 19.0, 10.2,
        "Formatted Input\nTitle + [SEP] + Abstract\n(Max 512 Tokens)",
        facecolor="#ffffff", edgecolor="#5d6d7e", fontsize=7.2,
    )
    _add_arrow(ax, (44.2, 31.7), (47.4, 31.7))

    _add_box(
        ax, 47.9, 26.6, 22.0, 10.2,
        "Asymmetric Doc Encoder\nallenai/specter2_base\n+ proximity Adapter",
        facecolor="#dceaf7", edgecolor="#2b5c8f", linewidth=1.05, fontsize=7.2,
    )
    _add_arrow(ax, (70.4, 31.7), (73.6, 31.7))

    _add_box(
        ax, 74.1, 26.6, 23.4, 10.2,
        "Document Matrix $\\mathbf{D}$\nLast-Layer [CLS] + L2 Norm\n(64,183 $\\times$ 768, float32)",
        facecolor="#e8f0fe", edgecolor="#1f4e78", linewidth=1.05, fontsize=7.2,
    )

    # === 2. Container Bawah: Online Dual-Input Retrieval Pipeline ===
    online_bg = FancyBboxPatch(
        (1.0, 1.0),
        98.0,
        21.2,
        boxstyle="round,pad=0.3,rounding_size=1.0",
        facecolor="#f7fbf9",
        edgecolor="#8ebfb3",
        linewidth=0.85,
        zorder=1,
    )
    ax.add_patch(online_bg)
    ax.text(
        2.4,
        20.6,
        "ONLINE DUAL-INPUT CONTROLLABLE RETRIEVAL PIPELINE  (Real-Time CPU Inference, 9.3 ms / Query)",
        ha="left",
        va="center",
        fontsize=7.3,
        fontweight="bold",
        color="#117a65",
        zorder=2,
    )

    # Dua kotak input terpisah (Q_pos dan Q_neg)
    _add_box(
        ax, 2.5, 11.4, 21.0, 7.0,
        "Positive Query ($Q_{\\mathrm{pos}}$)\nFlexible Research Topic",
        facecolor="#eaf7f0", edgecolor="#1e8449", linewidth=1.0, fontsize=7.2,
    )
    _add_box(
        ax, 2.5, 2.4, 21.0, 7.0,
        "Negative Preference ($Q_{\\mathrm{neg}}$)\nExcluded Concept (Optional)",
        facecolor="#fdf0ed", edgecolor="#c0392b", linewidth=1.0, fontsize=7.1,
    )

    _add_arrow(ax, (24.0, 14.9), (27.6, 13.6), color="#1e8449")
    _add_arrow(ax, (24.0, 5.9), (27.6, 7.2), color="#c0392b")

    # Shared Asymmetric Query Encoder
    _add_box(
        ax, 28.1, 3.0, 22.2, 14.8,
        "Shared Asymmetric\nQuery Encoder\nallenai/specter2_base\n+ adhoc_query Adapter\n([CLS] + L2 Norm)",
        facecolor="#e6f2ff", edgecolor="#1f618d", linewidth=1.05, fontsize=7.1,
    )

    # Dua panah keluar menuju Penalized Cosine Scoring (q_pos_hat & q_neg_hat)
    _add_arrow(ax, (50.8, 13.4), (55.8, 13.4), color="#1e8449")
    ax.text(53.3, 14.8, "$\\hat{\\mathbf{q}}_{\\mathrm{pos}}$", ha="center", va="center", fontsize=7.6, color="#1e8449")

    _add_arrow(ax, (50.8, 7.4), (55.8, 7.4), color="#c0392b")
    ax.text(53.3, 8.8, "$\\hat{\\mathbf{q}}_{\\mathrm{neg}}$", ha="center", va="center", fontsize=7.6, color="#c0392b")

    # Kotak Penalized Cosine Scoring (M3)
    _add_box(
        ax, 56.3, 3.0, 24.0, 14.8,
        "Penalized Cosine Scoring ($M_3$)\n"
        "$S_{\\mathrm{final}}(d) = \\hat{\\mathbf{q}}_{\\mathrm{pos}}\\!\\cdot\\!\\hat{\\mathbf{d}}"
        " - \\beta^*(\\hat{\\mathbf{q}}_{\\mathrm{neg}}\\!\\cdot\\!\\hat{\\mathbf{d}})$\n"
        "Calibrated $\\beta^* = 0.2$ (Val 20%)",
        facecolor="#fef9e7", edgecolor="#b7950b", linewidth=1.15, fontsize=7.1,
    )

    # Panah vertikal dari Document Matrix D turun ke Penalized Cosine Scoring pada x = 76.5
    _add_arrow(ax, (76.5, 26.1), (76.5, 18.3), color="#1f4e78", lw=1.15)
    ax.text(
        82.2,
        23.6,
        "$\\hat{\\mathbf{d}} \\in \\mathbf{D}$",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#1f4e78",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#ffffff", "edgecolor": "#9fb3c8", "lw": 0.6},
        zorder=6,
    )

    _add_arrow(ax, (80.8, 10.4), (83.8, 10.4), color="#117a65")

    # Kotak Output Top-K Recommendations & Metrics
    _add_box(
        ax, 84.3, 3.0, 13.2, 14.8,
        "Top-$K$ Ranked\nLiterature\n\nnDCG@10 ($\\uparrow$)\nRecall@10 ($\\uparrow$)\nVR@10 ($\\downarrow$)",
        facecolor="#eafaf1", edgecolor="#117a65", linewidth=1.05, fontsize=7.0,
    )

    fig.tight_layout(pad=0.05)
    png_path = figures_dir / "fig1_dual_input_architecture.png"
    pdf_path = figures_dir / "fig1_dual_input_architecture.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


if __name__ == "__main__":
    p_png, p_pdf = generate_fig1_architecture()
    print(f"[OK] Fig. 1 berhasil dibuat: {p_png} & {p_pdf}")
