"""
Generate efficient frontier, top ETF longs, and portfolio curve PNGs,
then stitch them into a one-page PDF report under data/plots/.

Assumptions:
- data/R_monthly.csv exists
- data/etfs_identifiers.txt exists (first 100 tickers are approved ETFs)

Outputs:
- data/plots/efficient_frontier.png
- data/plots/top10_etf_longs.png
- data/plots/portfolio_curve.png
- data/plots/allocation_report.pdf
"""
from __future__ import annotations

import os
import sys
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fortitudo.tech.optimization import MeanVariance


DATA_DIR = os.path.join(ROOT, "data")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")


def load_inputs():
    r_path = os.path.join(DATA_DIR, "R_monthly.csv")
    if not os.path.exists(r_path):
        raise FileNotFoundError(f"Missing {r_path}")
    Rdf = pd.read_csv(r_path, index_col=0, parse_dates=True)

    # Approved ETFs: first 100 tickers from etfs_identifiers.txt
    approved: list[str] = []
    etfs_path = os.path.join(DATA_DIR, "etfs_identifiers.txt")
    if os.path.exists(etfs_path):
        with open(etfs_path, "r", encoding="utf-8") as fh:
            for line in fh:
                m = re.match(r"\s*\d+\s+([A-Z]{2,5})\b", line)
                if m:
                    approved.append(m.group(1))
        approved = approved[:100]
    else:
        # Fallback: use any non-treasury columns
        approved = []

    cols = list(Rdf.columns)
    mask_treas = [c.startswith("DGS") or ("IRLTLT01" in c) for c in cols]
    etf_cols = [c for c, t in zip(cols, mask_treas) if not t]
    if approved:
        etf_cols = [c for c in etf_cols if c in set(approved)]
    if not etf_cols:
        etf_cols = [c for c, t in zip(cols, mask_treas) if not t][:100]

    treas_cols = [c for c, t in zip(cols, mask_treas) if t]
    keep_cols = etf_cols + treas_cols

    return Rdf, keep_cols, etf_cols


def classify_indices(keep_cols: list[str], etf_cols: list[str]):
    bond_etf = {
        # Broad agg / gov / corp / HY / munis / TIPS / duration buckets
        "AGG","BND","BNDX","BSV","CORP","EMB","EMLC","GOVT","HYG","HYLS","IEF",
        "IGSB","JNK","LOAN","LQD","MBB","MUB","MUNI","SHV","SHY","TIP","TIPS",
        "TLT","TR","TRST","USFR","VCIT","VCSH","VGSH","VTIP","JEPI","JPST"
    }
    # Treasuries series are bonds by definition
    equity_idx, bond_idx = [], []
    for i, c in enumerate(keep_cols):
        if c.startswith("DGS") or ("IRLTLT01" in c):
            bond_idx.append(i)
        elif c in etf_cols:
            if c in bond_etf:
                bond_idx.append(i)
            else:
                equity_idx.append(i)
        else:
            # If not recognized, treat as bond conservatively
            bond_idx.append(i)
    return equity_idx, bond_idx


def optimize(Rdf: pd.DataFrame, keep_cols: list[str], etf_cols: list[str], risk_cap_ann: float | None = 0.12):
    R = Rdf[keep_cols].values
    S, I = R.shape
    p = np.ones((S, 1)) / S
    mu = (p.T @ R).ravel()
    C = np.cov(R, rowvar=False, ddof=1) + 1e-4 * np.eye(I)

    # Constraints:
    # 1) Long-only: w >= 0  ->  -I w <= 0
    # 2) Per-asset cap: w <= 0.04  ->  I w <= 0.04
    # 3) Equity >= 50%: -sum(w_e) <= -0.5
    equity_idx, bond_idx = classify_indices(keep_cols, etf_cols)
    cons = []
    rhs = []
    # Long-only
    cons.append(-np.eye(I))
    rhs.append(np.zeros(I))
    # Per-asset cap 4%
    cons.append(np.eye(I))
    rhs.append(np.full(I, 0.04))
    # Equity >= 50%
    if equity_idx:
        row = np.zeros((1, I))
        row[0, equity_idx] = -1.0
        cons.append(row)
        rhs.append(np.array([-0.5]))
    G = np.vstack(cons)
    h = np.hstack(rhs)

    opt = MeanVariance(mean=mu, covariance_matrix=C, G=G, h=h)

    w_min = opt.efficient_portfolio().ravel()
    mean_min = float(mu @ w_min)
    frontier = []
    for k in range(120):
        tgt = mean_min + 0.002 * k
        try:
            w = opt.efficient_portfolio(return_target=tgt).ravel()
            m = float(mu @ w)
            v = float(np.sqrt(w @ C @ w))
            frontier.append((m, v, w))
        except Exception:
            break

    if not frontier:
        raise RuntimeError("Frontier construction failed; no feasible portfolios found.")

    means = np.array([m for m, _, _ in frontier])
    vols = np.array([v for _, v, _ in frontier])
    sharpe = np.divide(means, vols, out=np.full_like(means, np.nan), where=vols > 1e-12)

    # Apply optional annualized volatility cap when choosing best point
    best_i_all = int(np.nanargmax(sharpe))
    best_i = best_i_all
    if risk_cap_ann is not None and len(vols) > 0:
        cap_m = risk_cap_ann / np.sqrt(12.0)
        feas = np.where(vols <= cap_m)[0]
        if feas.size > 0:
            best_i = int(feas[np.nanargmax(sharpe[feas])])

    w = frontier[best_i][2]
    return means, vols, sharpe, best_i, w, equity_idx, bond_idx


def save_plots(Rdf: pd.DataFrame, keep_cols: list[str], etf_cols: list[str], means, vols, sharpe, best_i: int, w: np.ndarray):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Efficient frontier
    plt.figure(figsize=(7, 5))
    plt.plot(vols, means, "o-", alpha=0.6, label="Efficient frontier")
    plt.scatter([vols[best_i]], [means[best_i]], s=120, c="crimson", marker="*", label="Selected portfolio")
    plt.xlabel("Volatility (monthly)")
    plt.ylabel("Expected return (monthly)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    frontier_png = os.path.join(PLOTS_DIR, "efficient_frontier.png")
    plt.savefig(frontier_png, dpi=160)
    plt.close()

    # Top longs among ETFs
    w_series = pd.Series(w, index=keep_cols).sort_values(ascending=False)
    longs = w_series[w_series > 0]
    long_etfs = longs.loc[[c for c in longs.index if c in etf_cols]].head(10)
    plt.figure(figsize=(10, 6))
    ax = long_etfs.plot(kind="bar", color="#1f77b4")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Weight")
    ax.set_title("Top 10 ETF Long Weights (long-only, 4% max per asset)")
    plt.legend(["Weight"], loc="upper right")
    plt.tight_layout()
    top_png = os.path.join(PLOTS_DIR, "top10_etf_longs.png")
    plt.savefig(top_png, dpi=160)
    plt.close()

    # Portfolio curve for best Sharpe weights
    port_ret = (Rdf[keep_cols] @ w)
    curve = (1.0 + port_ret).cumprod()
    plt.figure(figsize=(9, 5))
    ax2 = curve.plot(label="Selected portfolio")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Index level (start=1)")
    plt.title("Portfolio Growth (Best Sharpe subject to caps)")
    plt.legend(loc="best")
    plt.tight_layout()
    curve_png = os.path.join(PLOTS_DIR, "portfolio_curve.png")
    plt.savefig(curve_png, dpi=160)
    plt.close()

    return frontier_png, top_png, curve_png, w_series, long_etfs


def save_pdf(frontier_png: str, top_png: str, curve_png: str, S: int, I: int, sharpe, best_i: int, means, vols, w_series: pd.Series, keep_cols: list[str], equity_idx: list[int], bond_idx: list[int], risk_cap_ann: float | None):
    pdf_path = os.path.join(PLOTS_DIR, "allocation_report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=landscape(letter))
    W, H = landscape(letter)
    margin = 24
    cell_w = (W - 3 * margin) / 2
    cell_h = (H - 3 * margin) / 2

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, H - margin + 4, "Allocation Report — Mean-Variance (Monthly, USD)")

    imgs = [frontier_png, curve_png, top_png]
    for idx, img in enumerate(imgs):
        r = idx // 2
        col = idx % 2
        x = margin + col * (cell_w + margin)
        y = H - margin - (r + 1) * (cell_h + margin)
        try:
            c.drawImage(ImageReader(img), x, y, width=cell_w, height=cell_h, preserveAspectRatio=True, anchor="sw")
        except Exception as e:
            c.setFont("Helvetica", 10)
            c.drawString(x, y + cell_h / 2, f"Error loading {os.path.basename(img)}: {e}")

    # Equity/Bond splits
    w_arr = w_series.reindex(keep_cols).values
    eq_w = float(np.sum(w_arr[equity_idx])) if equity_idx else 0.0
    bd_w = float(np.sum(w_arr[bond_idx])) if bond_idx else 0.0

    text_x = margin + (cell_w + margin)
    text_y = H - margin - cell_h + cell_h - 14
    c.setFont("Helvetica", 10)
    lines = [
        f"Scenarios: {S}  Instruments: {I}",
        f"Constraints: long-only, max 4% per asset; Equity ≥ 50% (actual ≈ {eq_w:.2f}), Bonds ≈ {bd_w:.2f}",
        f"Best Sharpe (m): {float(sharpe[best_i]):.3f}  |  Mean (m): {float(means[best_i]):.3%}  |  Vol (m): {float(vols[best_i]):.3%}",
        f"Best Sharpe (ann): {float(sharpe[best_i]*np.sqrt(12)):.3f}  |  Mean (ann): {(1+float(means[best_i]))**12-1:.3%}  |  Vol (ann): {float(vols[best_i]*np.sqrt(12)):.3%}",
    ]
    if risk_cap_ann is not None:
        lines.append(f"Risk cap: {risk_cap_ann:.0%} ann vol (applied when choosing best point)")
    for i, line in enumerate(lines):
        c.drawString(text_x, text_y - 14 * i, line)

    c.showPage()
    c.save()
    return pdf_path


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    Rdf, keep_cols, etf_cols = load_inputs()
    # Default risk cap 12% annualized; set to None to disable
    risk_cap_ann = float(os.environ.get("RISK_CAP_ANN", 0.12)) if os.environ.get("RISK_CAP_ANN") is not None else 0.12
    means, vols, sharpe, best_i, w, equity_idx, bond_idx = optimize(Rdf, keep_cols, etf_cols, risk_cap_ann=risk_cap_ann)
    frontier_png, top_png, curve_png, w_series, long_etfs = save_plots(Rdf, keep_cols, etf_cols, means, vols, sharpe, best_i, w)
    pdf_path = save_pdf(frontier_png, top_png, curve_png, Rdf.shape[0], len(keep_cols), sharpe, best_i, means, vols, w_series, keep_cols, equity_idx, bond_idx, risk_cap_ann)
    print("WROTE:")
    print(" ", frontier_png)
    print(" ", top_png)
    print(" ", curve_png)
    print(" ", pdf_path)


if __name__ == "__main__":
    main()
