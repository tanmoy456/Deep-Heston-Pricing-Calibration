import time
import QuantLib as ql

def heston_option_price(
        S0,         # Spot price
        K,          # Strike price
        r,          # Risk-free rate
        q,          # Dividend yield
        T,          # Time to maturity (in years)
        v0,         # Initial variance
        kappa,      # Mean reversion speed
        theta,      # Long-run variance
        sigma,      # Vol of vol
        rho,        # Correlation
        option_type="call"  # "call" or "put"
    ):
    # 1. Set evaluation date
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    # 2. Term structures
    day_count = ql.Actual365Fixed()
    risk_free_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, day_count)
    )
    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, q, day_count)
    )

    # 3. Spot handle
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))

    # 4. Heston process
    heston_process = ql.HestonProcess(
        risk_free_ts,
        dividend_ts,
        spot_handle,
        v0,
        kappa,
        theta,
        sigma,
        rho
    )

    # 5. Model + Engine
    heston_model = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(heston_model)

    # 6. European Option setup
    maturity_date = today + int(365.0 * T)

    if option_type.lower() == "call":
        ql_type = ql.Option.Call
    elif option_type.lower() == "put":
        ql_type = ql.Option.Put
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    payoff = ql.PlainVanillaPayoff(ql_type, K)
    exercise = ql.EuropeanExercise(maturity_date)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)

    # 7. Compute price
    return option.NPV()


# Example usage:
if __name__ == "__main__":
    
    m      = 0.5
    S0     = 4.0
    K      = m* S0
    r      = 0.05
    q      = 0.0
    T      = 1.0
    v0     = 0.05
    kappa  = 2.0
    theta  = 0.05
    sigma  = 0.3
    rho    = -0.5

    start = time.time()
    
    call_price = heston_option_price(S0, K, r, q, T, v0, kappa, theta, sigma, rho, option_type="call")
    put_price  = heston_option_price(S0, K, r, q, T, v0, kappa, theta, sigma, rho, option_type="put")
    
    end = time.time()
    print(f"Time taken = {end - start}")

    print(f"QuantLib Heston Analytic Call Price: {call_price:.8f}")
    print(f"QuantLib Heston Analytic Put Price: {put_price:.8f}")

    print(f"Call Price per unit spot: {call_price/K:.8f}")
    print(f"Put Price: {put_price/K:.8f}")

