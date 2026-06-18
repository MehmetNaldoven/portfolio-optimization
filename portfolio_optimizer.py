"""
Portfolio Optimizer — Modern Portfolio Theory
Pulls historical prices, finds max-Sharpe weights, plots the efficient frontier.

Usage:
    python portfolio_optimizer.py
"""


import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

# ── Configuration ──────────────────────────────────────────────────────────────

TICKERS        = ["AAPL"]   # Be sure that you have entered the tickers. Feel free to edit freely
START          = "2020-01-01"
END            = "2024-12-31"
RISK_FREE_RATE = 0.05    # annual risk-free rate (e.g. 5% T-bill)
N_SIMULATIONS  = 5_000   # random portfolios for the scatter plot

# ── Data ───────────────────────────────────────────────────────────────────────

def fetch_returns(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices via yfinance and return daily log returns."""
    max_retries = 4
    for attempt in range(1, max_retries + 1):
        print(f"Downloading price data (attempt {attempt}/{max_retries})...")
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]

        if not prices.empty:
            log_returns = np.log(prices / prices.shift(1)).dropna()
            print(f"Loaded {len(log_returns)} trading days for: {', '.join(tickers)}\n")
            return log_returns

        wait = attempt * 15
        print(f"  No data returned — waiting {wait}s before retry...")
        time.sleep(wait)

    raise ConnectionError(
        "Could not download price data after several attempts.\n"
        "Check your internet connection or try again in a few minutes."
    )

# ── Portfolio statistics ────────────────────────────────────────────────────────

def portfolio_stats(weights: np.ndarray, mean_returns: np.ndarray,
                    cov_matrix: np.ndarray, trading_days: int = 252):
    """Return (annualised return, annualised volatility, Sharpe ratio)."""
    ret  = np.dot(weights, mean_returns) * trading_days
    vol  = np.sqrt(weights @ cov_matrix @ weights)   # cov_matrix already annualised 
    sharpe = (ret - RISK_FREE_RATE) / vol
    return ret, vol, sharpe

# ── Optimisation ───────────────────────────────────────────────────────────────

def max_sharpe(mean_returns: np.ndarray, cov_matrix: np.ndarray) -> dict:
    n = len(mean_returns)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    init   = np.array([1 / n] * n)

    result = minimize(
        lambda w: -portfolio_stats(w, mean_returns, cov_matrix)[2],  
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    if not result.success:
        raise RuntimeError(f"Optimisation failed: {result.message}")

    weights = result.x
    ret, vol, sharpe = portfolio_stats(weights, mean_returns, cov_matrix)
    return {"weights": weights, "return": ret, "volatility": vol, "sharpe": sharpe}

def min_volatility(mean_returns: np.ndarray, cov_matrix: np.ndarray) -> dict:
    n = len(mean_returns)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    init   = np.array([1 / n] * n)

    result = minimize(
        lambda w: portfolio_stats(w, mean_returns, cov_matrix)[1],
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    weights = result.x
    ret, vol, sharpe = portfolio_stats(weights, mean_returns, cov_matrix)
    return {"weights": weights, "return": ret, "volatility": vol, "sharpe": sharpe}

def efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                        n_points: int = 100) -> tuple[list, list]:
    """Trace the efficient frontier across a range of target returns."""
    min_ret = mean_returns.min() * 252
    max_ret = mean_returns.max() * 252
    targets = np.linspace(min_ret, max_ret, n_points)

    n = len(mean_returns)
    frontier_vols, frontier_rets = [], []

    for target in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: portfolio_stats(w, mean_returns, cov_matrix)[0] - t},
        ]
        bounds = tuple((0, 1) for _ in range(n))
        result = minimize(
            lambda w: portfolio_stats(w, mean_returns, cov_matrix)[1],
            [1 / n] * n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if result.success:
            _, vol, _ = portfolio_stats(result.x, mean_returns, cov_matrix)
            frontier_vols.append(vol)
            frontier_rets.append(target)

    return frontier_vols, frontier_rets

# ── Monte Carlo scatter ─────────────────────────────────────────────────────────

def simulate_portfolios(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                         n: int = N_SIMULATIONS):
    num_assets = len(mean_returns)
    results = np.zeros((3, n))
    for i in range(n):
        w = np.random.dirichlet(np.ones(num_assets))
        r, v, s = portfolio_stats(w, mean_returns, cov_matrix)
        results[:, i] = [r, v, s]
    return results   # rows: return, vol, sharpe

# ── Reporting ──────────────────────────────────────────────────────────────────

def print_portfolio(label: str, portfolio: dict, tickers: list[str]):
    print(f"{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")
    print(f"  Expected annual return : {portfolio['return']:.2%}")
    print(f"  Annual volatility      : {portfolio['volatility']:.2%}")
    print(f"  Sharpe ratio           : {portfolio['sharpe']:.4f}")
    print(f"\n  Weights:")
    for ticker, weight in zip(tickers, portfolio["weights"]):
        bar = "█" * int(weight * 40)
        print(f"    {ticker:6s}  {weight:6.2%}  {bar}")
    print()

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log_returns = fetch_returns(TICKERS, START, END)
    mean_returns = log_returns.mean().values
    cov_matrix   = log_returns.cov().values * 252   # annualised

    # Optimise
    ms  = max_sharpe(mean_returns, cov_matrix)
    mv  = min_volatility(mean_returns, cov_matrix)

    print_portfolio("MAX SHARPE PORTFOLIO", ms, TICKERS)
    print_portfolio("MIN VOLATILITY PORTFOLIO", mv, TICKERS)

    # Monte Carlo simulation
    sim = simulate_portfolios(mean_returns, cov_matrix)

    # Efficient frontier
    ef_vols, ef_rets = efficient_frontier(mean_returns, cov_matrix)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#0f0f0f")

    scatter = ax.scatter(
        sim[1], sim[0], c=sim[2], cmap="plasma",
        alpha=0.5, s=6, label="Random portfolios"
    )
    plt.colorbar(scatter, ax=ax, label="Sharpe Ratio")

    ax.plot(ef_vols, ef_rets, "w--", linewidth=1.5, label="Efficient frontier")

    ax.scatter(ms["volatility"], ms["return"],
               marker="*", color="#00ff99", s=250, zorder=5, label="Max Sharpe")
    ax.scatter(mv["volatility"], mv["return"],
               marker="D", color="#ff4466", s=100, zorder=5, label="Min Volatility")

    ax.set_xlabel("Annualised Volatility", color="white")
    ax.set_ylabel("Annualised Return", color="white")
    ax.set_title("Efficient Frontier — Modern Portfolio Theory", color="white", fontsize=14)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    legend = ax.legend(facecolor="#1a1a1a", edgecolor="#444444", labelcolor="white")
    plt.tight_layout()
    plt.savefig("efficient_frontier.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Chart saved to efficient_frontier.png")


if __name__ == "__main__":
    main()
