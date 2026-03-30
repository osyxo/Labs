import numpy as np
import matplotlib.pyplot as plt

def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def dM_exact(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

def diff_central(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def run_lab():
    t0 = 1.0
    exact = dM_exact(t0)
    
    t_vals = np.linspace(0, 20, 500)
    plt.figure("Модель вологості ґрунту", figsize=(10, 6))
    plt.plot(t_vals, M(t_vals), label="M(t)")
    plt.title('Soil Moisture Model M(t)')
    plt.xlabel('t')
    plt.ylabel('M(t)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    h_values = np.logspace(-20, 2, 500)
    errors = [abs(diff_central(M, t0, h) - exact) for h in h_values]
    h0 = h_values[np.argmin(errors)]
    
    plt.figure("Аналіз похибки", figsize=(10, 6))
    plt.loglog(h_values, errors, label="Absolute Error")
    plt.axvline(h0, color='r', linestyle='--', label=f"h0 = {h0:.2e}")
    plt.title('Error Analysis (Log-Log Scale)')
    plt.xlabel('h')
    plt.ylabel('R')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()

    h = 1e-3
    y_h = diff_central(M, t0, h)
    y_2h = diff_central(M, t0, 2 * h)
    y_4h = diff_central(M, t0, 4 * h)
    
    y_rr = y_h + (y_h - y_2h) / 3
    y_e = (y_2h**2 - y_4h * y_h) / (2 * y_2h - (y_4h + y_h))
    p = (1 / np.log(2)) * np.log(abs((y_4h - y_2h) / (y_2h - y_h)))
    
    print(f"h0: {h0:.2e}")
    print(f"R1: {abs(y_h - exact):.10f}")
    print(f"y_RR: {y_rr:.10f}")
    print(f"R2: {abs(y_rr - exact):.10f}")
    print(f"y_E: {y_e:.10f}")
    print(f"p: {p:.2f}")
    print(f"R3: {abs(y_e - exact):.10f}")

if __name__ == "__main__":
    run_lab()