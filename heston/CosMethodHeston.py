import time
import numpy as np

# Chi and Psi helper functions for payoff coefficients
def Chi_Psi(a,b,c,d,k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a) / (b - a))
    if len(k) > 1:  # avoid division by zero for k=0
        psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c

    chi = (np.cos(k * np.pi * (d - a) / (b - a)) * np.exp(d) 
           - np.cos(k * np.pi * (c - a) / (b - a)) * np.exp(c)
           + (k * np.pi / (b - a)) * (np.sin(k * np.pi * (d - a) / (b - a)) 
           - np.sin(k * np.pi * (c - a) / (b - a)) * np.exp(c))) \
          / (1.0 + (k * np.pi / (b - a)) ** 2)
    return chi, psi

def CallPutCoefficients(CP,a,b,k):
    # if CP == OptionType.CALL:
    if CP.upper() == "CALL":
        c, d = 0.0, b
        chi_k, psi_k = Chi_Psi(a,b,c,d,k)
        if a < b and b < 0.0:
            H_k = np.zeros_like(k)
        else:
            H_k = 2.0/(b-a) * (chi_k - psi_k)
    # elif CP == OptionType.PUT:
    elif CP.upper() == "PUT":
        c, d = a, 0.0
        chi_k, psi_k = Chi_Psi(a,b,c,d,k)
        H_k = 2.0/(b-a) * (-chi_k + psi_k)
    return H_k

# Heston characteristic function
def ChFHestonModel(r, tau, kappa, sigma, vbar, v0, rho):

    # i = 1j
    i = complex(0.0,1.0)

    def cf(u):
        d = np.sqrt((kappa - sigma * rho * i * u)**2 +
                    (u**2 + i * u) * sigma**2)
        g = (kappa - sigma * rho * i * u - d) / (kappa - sigma * rho * i * u + d)

        A = (r * i * u * tau
             + (kappa * vbar * tau / sigma**2) * (kappa - sigma * rho * i * u - d)
             - (2 * kappa * vbar / sigma**2) * np.log((1 - g * np.exp(-d * tau)) / (1 - g)))

        C = ((1 - np.exp(-d * tau)) / (sigma**2 * (1 - g * np.exp(-d * tau)))
             * (kappa - sigma * rho * i * u - d))

        return np.exp(A + C * v0)

    return cf

# COS method for a single strike
def COS_Heston_Price(cf, CP, S0, r, tau, K, N=512, L=12):
    i = complex(0.0,1.0)
    x0 = np.log(S0 / K)
    
    # Truncation range
    a = -L * np.sqrt(tau)
    b =  L * np.sqrt(tau)
    
    k = np.arange(N)
    u = k * np.pi / (b - a)
    H_k = CallPutCoefficients(CP, a, b, k)
    
    mat = np.exp(i * (x0 - a) * u)
    temp = cf(u) * H_k
    temp[0] *= 0.5  # First term halved
    
    value = np.exp(-r * tau) * K * np.real(np.dot(mat, temp))
    return float(value)

# -------------------------
# Example usage
if __name__ == "__main__":
    # Heston params
    m     = 0.5       # moneyness
    S0    = 1.0
    K     = m*S0      # Strike price
    r     = 0.05
    tau   = 1.0
    kappa = 2.0
    sigma = 0.3
    vbar  = 0.05
    v0    = 0.05
    rho   = -0.5
    
    start = time.time()
    
    cf = ChFHestonModel(r, tau, kappa, sigma, vbar, v0, rho)

    call_price = COS_Heston_Price(cf, "call", S0, r, tau, K)
    put_price  = COS_Heston_Price(cf, "PUT",  S0, r, tau, K)

    end = time.time()

    print(f"Time taken = {end - start}")

    print("COS Call Price:", np.round(call_price,8))
    print("COS Put Price :", np.round(put_price, 8))

    print("European Call Option Price per unit spot:", np.round(call_price/K, 10))
    print("European Put Option Price per unit spot:", np.round(put_price/K, 10))
