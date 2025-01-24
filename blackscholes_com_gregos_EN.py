import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes function with Greeks calculation
def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    # Greeks common to calls and puts
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
             - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2))
    
    return price, delta, gamma, theta, vega, rho

# Interface configuration
st.set_page_config(layout="wide")
st.title("📊 Understanding the **Greeks** in the Black-Scholes Model")
st.markdown("Explore how parameters affect option price sensitivity (Delta, Gamma, Theta, Vega, Rho).")

# Sidebar with parameters
with st.sidebar:
    st.header("⚙️ Parameters")
    S = st.slider("Current Asset Price (S)", 50.0, 150.0, 100.0)
    K = st.slider("Strike Price (K)", 50.0, 150.0, 105.0)
    T = st.slider("Time to Expiry (years)", 0.1, 5.0, 1.0)
    r = st.slider("Risk-Free Rate (r)", 0.0, 0.2, 0.05)
    sigma = st.slider("Volatility (σ)", 0.1, 1.0, 0.2)
    option_type = st.radio("Option Type", ["call", "put"])

# Calculate price and Greeks
price, delta, gamma, theta, vega, rho = black_scholes_greeks(S, K, T, r, sigma, option_type)

# Display results in columns
col1, col2 = st.columns([1, 3])
with col1:
    st.success(f"### Option Price: **€{price:.2f}**")
    
    # Greeks table
    st.markdown("### Sensitivities (Greeks)")
    st.markdown(f"""
    - **Delta (Δ):** `{delta:.3f}`  
      *Change in option price per €1 change in the asset.*
    - **Gamma (Γ):** `{gamma:.3f}`  
      *Change in Delta per €1 change in the asset.*
    - **Theta (Θ):** `{theta:.3f}/day`  
      *Daily value erosion due to time decay.*
    - **Vega (ν):** `{vega:.3f}`  
      *Impact of a 1% increase in volatility.*
    - **Rho (ρ):** `{rho:.3f}`  
      *Impact of a 1% increase in interest rates.*
    """)

with col2:
    # Select Greek to visualize
    selected_greek = st.selectbox(
        "Select a Greek to visualize:",
        ["Delta", "Gamma", "Theta", "Vega", "Rho"],
        index=0
    )
    
    # Generate plot for selected Greek
    fig, ax = plt.subplots(figsize=(10, 5))
    S_range = np.linspace(50, 150, 100)
    
    # Calculate Greek values across S range
    greek_values = []
    for s in S_range:
        _, d, g, t, v, r_val = black_scholes_greeks(s, K, T, r, sigma, option_type)
        if selected_greek == "Delta":
            greek_values.append(d)
        elif selected_greek == "Gamma":
            greek_values.append(g)
        elif selected_greek == "Theta":
            greek_values.append(t / 365)  # Daily Theta
        elif selected_greek == "Vega":
            greek_values.append(v)
        else:
            greek_values.append(r_val)
    
    ax.plot(S_range, greek_values, color='darkorange', linewidth=2)
    ax.axvline(S, color='red', linestyle='--', label='Current Price (S)')
    ax.set_title(f"{selected_greek} vs Asset Price", fontweight='bold')
    ax.set_xlabel("Asset Price (S)")
    ax.set_ylabel(f"{selected_greek}")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)