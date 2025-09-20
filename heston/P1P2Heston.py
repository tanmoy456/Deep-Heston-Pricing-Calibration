import time
import numpy as np
from scipy import integrate

def heston_cf(phi, j, S0, v0, r, tau, kappa, theta, sigma, rho, lam=0.0):
    if j == 1:
        uj = 0.5
        bj = kappa + lam - rho * sigma
    elif j == 2:
        uj = -0.5
        bj = kappa + lam
    else:
        raise ValueError("j must be 1 or 2")
    
    i = 1j
    a = kappa * theta
    xi = bj - rho * sigma * i * phi
    d = np.sqrt((rho * sigma * i * phi - bj)**2 - sigma**2 * (2 * uj * i * phi - phi**2))
    g = (bj - rho * sigma * i * phi + d) / (bj - rho * sigma * i * phi - d)
    C = r * i * phi * tau + (a / sigma**2) * ((bj - rho * sigma * i * phi + d) * tau - 2.0 * np.log((1 - g * np.exp(d * tau)) / (1 - g)))
    D = (bj - rho * sigma * i * phi + d) / sigma**2 * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau)))
    x = np.log(S0)
    return np.exp(C + D * v0 + i * phi * x)

def P_j_integral(j, S0, v0, r, tau, K, kappa, theta, sigma, rho, lam=0.0):
    lnK = np.log(K)
    def integrand(phi):
        f = heston_cf(phi, j, S0, v0, r, tau, kappa, theta, sigma, rho, lam)
        val = np.real(np.exp(-1j * phi * lnK) * f / (1j * phi))
        return val
    result, _ = integrate.quad(integrand, 1e-8, 200.0, epsabs=1e-8, epsrel=1e-6, limit=200)
    return 0.5 + result / np.pi

def heston_call_price(S0, K, r, tau, kappa, theta, sigma, rho, v0, lam=0.0):
    P1 = P_j_integral(1, S0, v0, r, tau, K, kappa, theta, sigma, rho, lam)
    P2 = P_j_integral(2, S0, v0, r, tau, K, kappa, theta, sigma, rho, lam)
    return S0 * P1 - K * np.exp(-r * tau) * P2

def heston_put_price(S0, K, r, tau, kappa, theta, sigma, rho, v0, lam=0.0):
    # Put-call parity: P = C - S0 + K e^{-r tau}
    call = heston_call_price(S0, K, r, tau, kappa, theta, sigma, rho, v0, lam)
    return call - S0 + K * np.exp(-r * tau)

if __name__ == "__main__":
    
    # Given parameters
    m = 0.5       # moneyness
    S0 = 1        # Initial stock price (spot price)
    K = m*S0      # Strike price
    r = 0.05      # Risk-free rate
    T = 1.0       # Time to maturity
    kappa = 2.0   # Mean reversion rate
    theta = 0.05  # Long-term average volatility
    sigma = 0.3   # Volatility of volatility
    rho = -0.5    # Correlation coefficient
    v0 = 0.05     # Initial volatility

    start = time.time()
    call_price = heston_call_price(S0, K, r, T, kappa, theta, sigma, rho, v0)
    put_price = heston_put_price(S0, K, r, T, kappa, theta, sigma, rho, v0)
    end = time.time()

    print(f"Time taken = {end - start}")
    
    print("European Call Option Price:", np.round(call_price, 10))
    print("European Put Option Price:", np.round(put_price, 10))

    print("European Call Option Price per unit spot:", np.round(call_price/K, 10))
    print("European Put Option Price per unit spot:", np.round(put_price/K, 10))
