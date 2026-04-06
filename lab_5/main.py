import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

a, b = 0, 24

x_sym = sp.Symbol('x')
f_sym = 50 + 20 * sp.sin(sp.pi * x_sym / 12) + 5 * sp.exp(-0.2 * (x_sym - 12)**2)
I0 = float(sp.integrate(f_sym, (x_sym, a, b)).evalf())

def simpson(f, a, b, N):
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    return (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])

N_vals = np.arange(10, 1002, 2)
errors = []
N_opt = None
eps_opt = None

for N in N_vals:
    I_N = simpson(f, a, b, N)
    err = abs(I_N - I0)
    errors.append(err)
    if err <= 1e-12 and N_opt is None:
        N_opt = N
        eps_opt = err

if N_opt is None:
    N_opt = N_vals[-1]
    eps_opt = errors[-1]

N0 = max(8, int((N_opt / 10) // 8 * 8))
if N0 % 8 != 0: 
    N0 = 8

I_N0 = simpson(f, a, b, N0)
eps0 = abs(I_N0 - I0)

I_N0_2 = simpson(f, a, b, N0 // 2)
I_R = I_N0 + (I_N0 - I_N0_2) / 15
epsR = abs(I_R - I0)

I_N0_4 = simpson(f, a, b, N0 // 4)
denominator = 2 * I_N0_2 - (I_N0 + I_N0_4)
if denominator != 0:
    I_E = (I_N0_2**2 - I_N0 * I_N0_4) / denominator
    p = (1 / np.log(2)) * np.log(abs((I_N0_4 - I_N0_2) / (I_N0_2 - I_N0)))
else:
    I_E = I_N0
    p = 0
epsE = abs(I_E - I0)

eval_count = 0

def adaptive_simpson(f, a, b, tol):
    global eval_count
    c = (a + b) / 2
    h = b - a
    fa, fb, fc = f(a), f(b), f(c)
    eval_count += 3
    S1 = (h / 6) * (fa + 4 * fc + fb)
    return _adaptive_simpson_recursive(f, a, b, tol, S1, fa, fb, fc)

def _adaptive_simpson_recursive(f, a, b, tol, S1, fa, fb, fc):
    global eval_count
    c = (a + b) / 2
    d = (a + c) / 2
    e = (c + b) / 2
    fd, fe = f(d), f(e)
    eval_count += 2
    h = b - a
    S_left = (h / 12) * (fa + 4 * fd + fc)
    S_right = (h / 12) * (fc + 4 * fe + fb)
    S2 = S_left + S_right
    if abs(S1 - S2) <= 15 * tol:
        return S2 + (S2 - S1) / 15
    return _adaptive_simpson_recursive(f, a, c, tol / 2, S_left, fa, fc, fd) + \
           _adaptive_simpson_recursive(f, c, b, tol / 2, S_right, fc, fb, fe)

tol_vals = [1e-4, 1e-6, 1e-8, 1e-10]
adapt_results = []
for tol in tol_vals:
    eval_count = 0
    I_adapt = adaptive_simpson(f, a, b, tol)
    err_adapt = abs(I_adapt - I0)
    adapt_results.append((tol, I_adapt, err_adapt, eval_count))

print(f"Точне значення інтегралу I0: {I0}")
print(f"N_opt для заданої точності 1e-12: {N_opt}, eps_opt: {eps_opt}")
print(f"N0: {N0}, eps0: {eps0}")
print(f"Рунге-Ромберг: I_R = {I_R}, epsR = {epsR}")
print(f"Ейткен: I_E = {I_E}, epsE = {epsE}, Порядок методу p = {p}")
print("Адаптивний алгоритм (tol, I_adapt, err_adapt, eval_count):")
for res in adapt_results:
    print(res)

plt.figure(figsize=(10, 6))
plt.plot(N_vals, errors, label='Похибка Simpson(N)')
plt.axhline(1e-12, color='r', linestyle='--', label='Задана точність 1e-12')
if N_opt is not None:
    plt.plot(N_opt, eps_opt, 'ro', label=f'N_opt = {N_opt}')
plt.yscale('log')
plt.xlabel('N (Кількість розбиттів)')
plt.ylabel('Похибка |I(N) - I0|')
plt.title('Залежність точності обчислення інтегралу від N')
plt.legend()
plt.grid(True)
plt.show()

tols = [r[0] for r in adapt_results]
evals = [r[3] for r in adapt_results]
errs_ad = [r[2] for r in adapt_results]

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('Tolerance (tol)')
ax1.set_ylabel('Похибка', color='tab:blue')
ax1.plot(tols, errs_ad, 'bo-', label='Похибка')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xscale('log')
ax1.set_yscale('log')

ax2 = ax1.twinx()
ax2.set_ylabel('Кількість обчислень f(x)', color='tab:red')
ax2.plot(tols, evals, 'rs-', label='Виклики функції')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Аналіз адаптивного алгоритму')
fig.tight_layout()
plt.grid(True)
plt.show()