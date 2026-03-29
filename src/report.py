from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_nav(port: pd.DataFrame, save_path: Path | None = None) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(port["date"], port["gross_nav"], label="Gross NAV")
    plt.plot(port["date"], port["net_nav"], label="Net NAV")
    plt.legend()
    plt.title("Portfolio NAV")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_turnover(port: pd.DataFrame, save_path: Path | None = None) -> None:
    plt.figure(figsize=(9, 4))
    plt.plot(port["date"], port["turnover"])
    plt.title("Portfolio Turnover")
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rank_ic(rank_ic: pd.DataFrame, save_path: Path | None = None) -> None:
    plt.figure(figsize=(9, 4))
    plt.plot(rank_ic["date"], rank_ic["rank_ic"])
    plt.axhline(rank_ic["rank_ic"].mean(), linestyle="--")
    plt.title("Rank IC by Date")
    plt.xlabel("Date")
    plt.ylabel("Rank IC")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_feature_importance(fi: pd.DataFrame, save_path: Path | None = None, topn: int = 20) -> None:
    fi = fi.head(topn).copy()
    plot_col = "importance" if "importance" in fi.columns else fi.columns[-1]
    plt.figure(figsize=(8, 6))
    plt.barh(fi["feature"][::-1], fi[plot_col][::-1])
    plt.title(f"Top {topn} Feature Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
