import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from heston.QuantlibHeston import heston_option_price
from BlackScholes.BSprice import implied_volatility

# --- Set seeds for reproducibility ---
keras.backend.set_floatx('float32')
tf.random.set_seed(1234)
np.random.seed(1234)

# --- Load dataset and model with caching ---
@st.cache_resource
def load_data_and_model(mode):
    if mode == 'PRICE':
        dataset = np.load('heston_data/Heston_ql_LHS_Feller_CALL_r_0.0_T_K_20k.npy')
        model = keras.models.load_model(
            'Heston_ql_CALL_full_model.h5',
            custom_objects={'root_mean_squared_error': root_mean_squared_error})
    else:
        dataset = np.load('heston_data/Heston_ql_LHS_Feller_IV_r_0.0_T_K_20k.npy')
        model = keras.models.load_model(
            'Heston_ql_IV_full_model.h5',
            custom_objects={'root_mean_squared_error': root_mean_squared_error})

    X_all = dataset[:, :5]
    y_all = dataset[:, 5:]

    # Train/test split (not used for training here, but needed for scaler)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=1234)

    # Input scaling params
    X_lb = X_all.min(axis=0)
    X_ub = X_all.max(axis=0)
    X_mid = 0.5 * (X_ub + X_lb)
    X_span = (X_ub - X_lb)

    # Output scaling with StandardScaler on train labels only
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)

    return model, X_mid, X_span, y_scaler

# --- Scaling functions ---
def scale_X(x, X_mid, X_span):
    return (x - X_mid) * (2.0 / X_span)

def inverse_y(y_scaled, scaler):
    return scaler.inverse_transform(y_scaled)

# --- Custom RMSE loss ---
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# --- Functions for price and IV generation ---
def generate_price(S0, r, strikes, maturities, true_params, option_type="call"):
    market_prices = []
    for T in maturities:
        row = []
        for K in strikes:
            price = heston_option_price(
                S0, K, r, 0.0, T,
                true_params["v0"],
                true_params["kappa"],
                true_params["theta"],
                true_params["sigma"],
                true_params["rho"],
                option_type=option_type)
            row.append(price)
        market_prices.append(row)
    return np.array(market_prices)

def generate_iv(S0, r, strikes, maturities, true_params, option_type="call"):
    market_ivs = []
    for T in maturities:
        row = []
        for K in strikes:
            price = heston_option_price(
                S0, K, r, 0.0, T,
                true_params["v0"],
                true_params["kappa"],
                true_params["theta"],
                true_params["sigma"],
                true_params["rho"],
                option_type=option_type
            )
            iv = implied_volatility(price, S0, K, T, r, q=0, option_type=option_type)
            row.append(iv)
        market_ivs.append(row)
    return np.array(market_ivs)

def check_feller(params):
    kappa = params["kappa"]
    theta = params["theta"]
    sigma = params["sigma"]
    fel_cond = 2 * kappa * theta / (sigma ** 2)
    return fel_cond

# --- Comparison and plotting ---
def compare_and_plot(S0, r, strikes, maturities, custom_params, model, X_mid, X_span, y_scaler, mode):

    strikes_dim = len(strikes)
    maturities_dim = len(maturities)

    if mode == 'PRICE':
        true_values = generate_price(S0, r, strikes, maturities, custom_params, option_type='call')
        ylabel = "Call Price"
    else:
        true_values = generate_iv(S0, r, strikes, maturities, custom_params, option_type='call')
        ylabel = "Implied Volatility"

    # Prepare model input vector in order: v0, rho, sigma, theta, kappa
    param_vector = np.array([
        custom_params["v0"],
        custom_params["rho"],
        custom_params["sigma"],
        custom_params["theta"],
        custom_params["kappa"]
    ]).reshape(1, -1)

    # Scale inputs
    param_vector_scaled = scale_X(param_vector, X_mid, X_span)

    # Predict and inverse scale
    pred_scaled = model.predict(param_vector_scaled, verbose=0)
    pred_values = inverse_y(pred_scaled, y_scaler)[0].reshape(maturities_dim, strikes_dim)

    # Plot results
    fig, axes = plt.subplots(3, 3, figsize=(9, 9),  dpi=200, constrained_layout=True)

    axes = axes.flatten()

    for i, T in enumerate(maturities):
        ax = axes[i]
        ax.plot(strikes / S0, true_values[i], label='True', marker='o')
        ax.plot(strikes / S0, pred_values[i], '--r', label='NN Predicted', marker='x')
        ax.set_title(f'Maturity = {T:.2f}')
        ax.set_xlabel('K/S0')
        ax.set_ylabel(ylabel)
        ax.legend()

    # Hide extra subplots if any
    for j in range(len(maturities), len(axes)):
        axes[j].axis('off')

    rmse = np.sqrt(np.mean((pred_values - true_values)**2))
    # st.pyplot(fig)
    st.pyplot(fig, use_container_width=True)
    # st.write(f"RMSE between true and predicted surfaces: {rmse:.6f}")

def compare_and_plot_plotly(S0, r, strikes, maturities, custom_params, model, X_mid, X_span, y_scaler, mode):

    strikes_dim = len(strikes)
    maturities_dim = len(maturities)

    if mode == 'PRICE':
        true_values = generate_price(S0, r, strikes, maturities, custom_params, option_type='call')
        ylabel = "Call Price"
    else:
        true_values = generate_iv(S0, r, strikes, maturities, custom_params, option_type='call')
        ylabel = "Implied Volatility"

    param_vector = np.array([
        custom_params["v0"],
        custom_params["rho"],
        custom_params["sigma"],
        custom_params["theta"],
        custom_params["kappa"]
    ]).reshape(1, -1)

    param_vector_scaled = scale_X(param_vector, X_mid, X_span)
    pred_scaled = model.predict(param_vector_scaled, verbose=0)
    pred_values = inverse_y(pred_scaled, y_scaler)[0].reshape(maturities_dim, strikes_dim)

    rmse = np.sqrt(np.mean((pred_values - true_values) ** 2))

    # --- simple Plotly plotting ---
    rows = 3
    cols = 3
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"Maturity = {T:.2f}" for T in maturities])

    true_color = 'blue'
    pred_color = 'red'

    for i, T in enumerate(maturities):
        row = (i // cols) + 1
        col = (i % cols) + 1
        x = strikes / S0

        # True (blue) — name given for all traces, but show legend only for first occurrence
        fig.add_trace(
            go.Scatter(
                x=x, y=true_values[i],
                mode='lines+markers',
                name='True',
                showlegend=(i == 0),               # <= important: show legend only once
                marker=dict(symbol='circle', size=6, color=true_color),
                line=dict(color=true_color),
                hovertemplate='K/S0=%{x:.3f}<br>True=%{y:.6f}<extra></extra>'
            ),
            row=row, col=col
        )

        # Predicted (red)
        fig.add_trace(
            go.Scatter(
                x=x, y=pred_values[i],
                mode='lines+markers',
                name='NN Predicted',
                showlegend=(i == 0),               # <= only show once
                marker=dict(symbol='x', size=6, color=pred_color),
                line=dict(color=pred_color, dash='dash'),
                hovertemplate='K/S0=%{x:.3f}<br>NN=%{y:.6f}<extra></extra>'
            ),
            row=row, col=col
        )

        # axis titles on the edges
        if row == rows:
            fig.update_xaxes(title_text='K / S0', row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text=ylabel, row=row, col=col)

    # hide unused subplots
    total = rows * cols
    if maturities_dim < total:
        for j in range(maturities_dim, total):
            r = (j // cols) + 1
            c = (j % cols) + 1
            fig.update_xaxes(visible=False, row=r, col=c)
            fig.update_yaxes(visible=False, row=r, col=c)

    # legend at top, horizontal (only two entries now)
    fig.update_layout(
        height=900,
        width=1000,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(t=80, b=40, l=60, r=20)
    )

    st.plotly_chart(fig, use_container_width=True)


# --- Streamlit UI ---

# st.title("("Neural Network Pricer & Implied Volatility Visualizer")
# st.header("Neural Network Pricer & Implied Volatility Visualizer")
st.subheader("Neural Network Pricer & Implied Volatility Visualizer")

# Select mode
# mode = st.radio("Select Mode", options=["PRICE", "IV"])
mode = st.sidebar.radio("Select Mode", options=["PRICE", "IV"])

# Load model, data, scalers once
modelGEN, X_mid, X_span, y_scaler = load_data_and_model(mode)

st.sidebar.header("Heston Parameters")

v0 = st.sidebar.slider('$v_0$', 0.0001, 0.04, 0.01, 0.001, format="%.4f")
rho = st.sidebar.slider('$\\rho$', -1.0, -0.1, -0.92, 0.01, format="%.2f")
sigma = st.sidebar.slider('$\\sigma$', 0.01, 1.0, 0.895, 0.01)
theta = st.sidebar.slider('$\\theta$', 0.01, 0.2, 0.12, 0.01)
kappa = st.sidebar.slider('$\\kappa$', 1.0, 10.0, 6.81, 0.01)

custom_params = {'kappa': kappa, 'theta': theta, 'sigma': sigma, 'rho': rho, 'v0': v0}

fel_cond = check_feller(custom_params)

# Render the math expression with the numeric value
st.sidebar.latex(rf"\dfrac{{2\kappa\theta}}{{\sigma^2}} = {fel_cond:.3f}")

if fel_cond >= 1.0:
    st.sidebar.success("Feller condition HOLDS")
else:
    st.sidebar.error("Feller condition VIOLATED")

# Fixed constants
S0 = 1.0
r = 0.0
# strikes and maturities
strikes = np.array([0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5])
maturities = np.array([0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.0])

compare_and_plot(S0, r, strikes, maturities, custom_params, modelGEN, X_mid, X_span, y_scaler, mode)

# compare_and_plot_plotly(S0, r, strikes, maturities, custom_params, modelGEN, X_mid, X_span, y_scaler, mode)

st.info(f"Fixed parameters: Spot Price $S_0$ = {S0}, Risk-Free Rate $r$ = {r}")
