import time
import numpy as np

# ===============================
#   Heston Monte Carlo Simulation
# ===============================

def GeneratePathsHestonEuler(NoOfPaths, NoOfSteps, T, r, S0, kappa, gamma, rho, vbar, v0, seed=None):
    """Simulate Heston model paths using Euler-Maruyama scheme."""
    if seed is not None:
        np.random.seed(seed)
        
    Z1 = np.random.normal(0.0, 1.0, (NoOfPaths, NoOfSteps))
    Z2 = np.random.normal(0.0, 1.0, (NoOfPaths, NoOfSteps))
    
    W1 = np.zeros((NoOfPaths, NoOfSteps+1))
    W2 = np.zeros((NoOfPaths, NoOfSteps+1))
    V  = np.zeros((NoOfPaths, NoOfSteps+1))
    X  = np.zeros((NoOfPaths, NoOfSteps+1))
    
    V[:,0] = v0
    X[:,0] = np.log(S0)
    time_grid = np.zeros(NoOfSteps+1)
    dt = T / NoOfSteps
    
    for i in range(NoOfSteps):
        if NoOfPaths > 1:
            Z1[:,i] = (Z1[:,i] - Z1[:,i].mean()) / Z1[:,i].std()
            Z2[:,i] = (Z2[:,i] - Z2[:,i].mean()) / Z2[:,i].std()
        Z2[:,i] = rho * Z1[:,i] + np.sqrt(1.0 - rho**2) * Z2[:,i]
        
        W1[:,i+1] = W1[:,i] + np.sqrt(dt) * Z1[:,i]
        W2[:,i+1] = W2[:,i] + np.sqrt(dt) * Z2[:,i]
        
        V[:,i+1] = V[:,i] + kappa*(vbar - V[:,i]) * dt + gamma * np.sqrt(np.maximum(V[:,i], 0)) * (W1[:,i+1] - W1[:,i])
        V[:,i+1] = np.maximum(V[:,i+1], 0.0)
        
        X[:,i+1] = X[:,i] + (r - 0.5*V[:,i])*dt + np.sqrt(np.maximum(V[:,i], 0)) * (W2[:,i+1] - W2[:,i])
        time_grid[i+1] = time_grid[i] + dt
    
    return {"time": time_grid, "S": np.exp(X)}

def CIR_Sample(NoOfPaths, kappa, gamma, vbar, s, t, v_s):
    """Exact sampling from CIR process for variance in AES scheme."""
    delta = 4.0 * kappa * vbar / gamma**2
    c = (gamma**2) / (4.0 * kappa) * (1.0 - np.exp(-kappa*(t-s)))
    kappa_bar = 4.0 * kappa * v_s * np.exp(-kappa*(t-s)) / (gamma**2 * (1.0 - np.exp(-kappa*(t-s))))
    return c * np.random.noncentral_chisquare(delta, kappa_bar, NoOfPaths)

def GeneratePathsHestonAES(NoOfPaths, NoOfSteps, T, r, S0, kappa, gamma, rho, vbar, v0, seed=None):
    """Simulate Heston model paths using Almost Exact Scheme."""
    if seed is not None:
        np.random.seed(seed)

    Z1 = np.random.normal(0.0, 1.0, (NoOfPaths, NoOfSteps))
    W1 = np.zeros((NoOfPaths, NoOfSteps+1))
    V  = np.zeros((NoOfPaths, NoOfSteps+1))
    X  = np.zeros((NoOfPaths, NoOfSteps+1))
    V[:,0] = v0
    X[:,0] = np.log(S0)
    time_grid = np.zeros(NoOfSteps+1)
    dt = T / NoOfSteps
    
    for i in range(NoOfSteps):
        if NoOfPaths > 1:
            Z1[:,i] = (Z1[:,i] - Z1[:,i].mean()) / Z1[:,i].std()
        W1[:,i+1] = W1[:,i] + np.sqrt(dt) * Z1[:,i]
        
        V[:,i+1] = CIR_Sample(NoOfPaths, kappa, gamma, vbar, 0, dt, V[:,i])
        
        k0 = (r - rho/gamma * kappa * vbar) * dt
        k1 = (rho * kappa/gamma - 0.5) * dt - rho / gamma
        k2 = rho / gamma
        
        X[:,i+1] = X[:,i] + k0 + k1 * V[:,i] + k2 * V[:,i+1] \
                   + np.sqrt((1.0 - rho**2) * V[:,i]) * (W1[:,i+1] - W1[:,i])
        time_grid[i+1] = time_grid[i] + dt
    
    return {"time": time_grid, "S": np.exp(X)}

# ===============================
#   Payoff and Monte Carlo Price
# ===============================

def price_option_mc(option_type, S_paths, K, r, T):
    """
    Compute option price from simulated paths via payoff expectation.
    option_type: 'call' or 'put'
    """
    S_T = S_paths[:, -1]
    
    if option_type.lower() == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)
    
    return np.exp(-r * T) * np.mean(payoffs)

# ===============================
#   Example Usage
# ===============================

if __name__ == "__main__":
    # Parameters
    NoOfPaths = 10000
    NoOfSteps = 350
    T     = 1.0
    r     = 0.05
    kappa = 2.0
    gamma = 0.3
    rho   = -0.5
    vbar  = 0.05
    v0    = 0.05
    m     = 0.5 
    S0    = 4
    K     = m*S0

    start_time = time.time()
    
    # Euler MC
    euler_paths = GeneratePathsHestonEuler(NoOfPaths, NoOfSteps, T, r, S0, kappa, gamma, rho, vbar, v0, seed=42)
    euler_call_price = price_option_mc("call", euler_paths["S"], K, r, T)
    euler_put_price  = price_option_mc("put",  euler_paths["S"], K, r, T)

    # AES MC
    aes_paths = GeneratePathsHestonAES(NoOfPaths, NoOfSteps, T, r, S0, kappa, gamma, rho, vbar, v0, seed=42)
    aes_call_price = price_option_mc("call", aes_paths["S"], K, r, T)
    aes_put_price  = price_option_mc("put",  aes_paths["S"], K, r, T)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Euler MC Call Price: {euler_call_price:.8f}")
    print(f"Euler MC Put Price : {euler_put_price:.8f}")
    print(f"AES   MC Call Price: {aes_call_price:.8f}")
    print(f"AES   MC Put Price : {aes_put_price:.8f}")
    print(f"Total runtime: {elapsed_time:.4f} seconds")

    print(f"Euler MC Call Price per unit spot: {euler_call_price/K:.8f}")
    print(f"Euler MC Put Price per unit spot : {euler_put_price/K:.8f}")
