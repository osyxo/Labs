import numpy as np
import matplotlib.pyplot as plt

months = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
temps = np.array([-2, 0, 5, 10, 15, 20, 23, 22, 17, 10, 5, 0, -10, 3, 7, 13, 19, 20, 22, 21, 18, 15, 10, 3])

def form_matrix_b(x, m):
    matrix = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for l in range(m + 1):
            matrix[k, l] = np.sum(x**(k + l))
    return matrix

def form_vector_c(x, y, m):
    vector = np.zeros(m + 1)
    for k in range(m + 1):
        vector[k] = np.sum(y * (x**k))
    return vector

def gauss_solve(A, b):
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)
    
    for k in range(n):
        max_row = k + np.argmax(np.abs(A[k:, k]))
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]
        
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
            
    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.dot(A[i, i + 1:], x_sol[i + 1:])) / A[i, i]
    return x_sol

def calculate_polynomial(x, coef):
    y_poly = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coef):
        y_poly += c * (x**i)
    return y_poly

def calculate_variance(y_true, y_approx):
    n = len(y_true)
    return np.sqrt(np.sum((y_approx - y_true)**2) / n)

variances = []
degrees = range(1, 11)

for m in degrees:
    B = form_matrix_b(months, m)
    C = form_vector_c(months, temps, m)
    coeffs = gauss_solve(B, C)
    approx = calculate_polynomial(months, coeffs)
    var = calculate_variance(temps, approx)
    variances.append(var)
    print(f"m={m}, var={var:.4f}")

optimal_m = degrees[np.argmin(variances)]
final_B = form_matrix_b(months, optimal_m)
final_C = form_vector_c(months, temps, optimal_m)
final_coeffs = gauss_solve(final_B, final_C)

future_months = np.array([25, 26, 27])
forecast = calculate_polynomial(future_months, final_coeffs)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.scatter(months, temps, color='red')
x_fine = np.linspace(1, 27, 200)
y_fine = calculate_polynomial(x_fine, final_coeffs)
plt.plot(x_fine, y_fine)
plt.scatter(future_months, forecast, color='green')

plt.subplot(2, 1, 2)
errors = np.abs(temps - calculate_polynomial(months, final_coeffs))
plt.bar(months, errors)
plt.show()