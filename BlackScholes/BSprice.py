import time
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

### ---- Black-Scholes price for a European option ----
def bs_option_price(S, K, T, r, sigma, q=0, option_type="call"):
    """
    Computes the Black-Scholes price for a European call or put option.
    Parameters:
    S           : float : Spot price of the underlying asset
    K           : float : Strike price
    T           : float : Time to expiration (years)
    r           : float : Risk-free interest rate (as decimal, e.g., 0.05)
    sigma       : float : Volatility (annualized standard deviation)
    q           : float : Continuous dividend yield (as decimal, optional, default 0)
    option_type : str   : 'call' (default) or 'put'
    Returns:
    price : float : Theoretical fair value of the specified option type
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price

### --- Implied volatility Function ---
def implied_volatility(market_price, S, K, T, r, q=0, option_type='call'):
    """
    Implied volatility using Black-Scholes, for either a call or put.
    """
    if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
        return np.nan

    def objective_function(sigma):
        return bs_option_price(S, K, T, r, sigma, q, option_type) - market_price

    try:
        implied_vol = brentq(objective_function, 1e-6, 5)
    except (ValueError, RuntimeError):
        implied_vol = np.nan
    return implied_vol


if __name__ == "__main__":
    # Test case for call option
    m = 0.5
    S = 5          # Spot price
    K = m*S        # Strike price
    T = 1.0        # Time to maturity 1 year
    r = 0.05       # Risk-free rate 5%
    q = 0.0       # Dividend yield 2%
    sigma_true = 0.25  # True volatility

    # Calculate theoretical price
    print("True Volatility: ", sigma_true )
    call_price = bs_option_price(S, K, T, r, sigma_true, q, option_type="call")
    print(f"Black-Scholes Call Price: {call_price:.10f}")

    # Now recover implied volatility from the call price
    implied_vol = implied_volatility(call_price, S, K, T, r, q, option_type="call")
    print(f"Implied Volatility from Call Price: {implied_vol:.4f}")

    # Test case for put option
    put_price = bs_option_price(S, K, T, r, sigma_true, q, option_type="put")
    print(f"Black-Scholes Put Price: {put_price:.10f}")

    implied_vol_put = implied_volatility(put_price, S, K, T, r, q, option_type="put")
    print(f"Implied Volatility from Put Price: {implied_vol_put:.4f}")

    print(f"Black-Scholes Call Price per unit spot: {call_price/S:.10f}")
    print(f"Black-Scholes Put Price per unit spot: {put_price/S:.10f}")
