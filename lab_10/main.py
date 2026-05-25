import numpy as np
import matplotlib.pyplot as plt

# Базова функція f(x, y) = dy/dx
def f(x, y):
    return y - x**2

# Точний аналітичний розв'язок
def y_exact(x):
    return x**2 + 2*x + 2 - np.exp(x)

# =====================================================================
# МЕТОД РУНГЕ-КУТТА 4-ГО ПОРЯДКУ
# =====================================================================
def runge_kutta_4_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + h*k1/2)
    k3 = f(x + h/2, y + h*k2/2)
    k4 = f(x + h, y + h*k3)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def runge_kutta_4_fixed(x0, y0, x_max, h):
    x_vals = [x0]
    y_vals = [y0]
    x = x0
    y = y0
    while x < x_max - 1e-10:
        if x + h > x_max:
            h = x_max - x
        y = runge_kutta_4_step(x, y, h)
        x += h
        x_vals.append(x)
        y_vals.append(y)
    return np.array(x_vals), np.array(y_vals)

def runge_kutta_4_adaptive(x0, y0, x_max, eps):
    x_vals = [x0]
    y_vals = [y0]
    h_vals = []
    
    x = x0
    y = y0
    h = 0.1  # Початковий крок
    
    while x < x_max - 1e-10:
        if x + h > x_max:
            h = x_max - x
            
        # Один крок довжиною h
        y_single = runge_kutta_4_step(x, y, h)
        # Два кроки довжиною h/2
        y_half1 = runge_kutta_4_step(x, y, h/2)
        y_double = runge_kutta_4_step(x + h/2, y_half1, h/2)
        
        # Оцінка похибки за методом Рунге
        error = (16.0 / 15.0) * abs(y_single - y_double)
        
        if error > eps:
            h /= 2  # Зменшуємо крок
            continue
        else:
            h_vals.append(h)
            y = y_double
            x += h
            x_vals.append(x)
            y_vals.append(y)
            
            if error < eps / 64:  # Поріг збільшення кроку (2^(s+1) = 32 або 64)
                h *= 2
                
    return np.array(x_vals), np.array(y_vals), h_vals

# =====================================================================
# МЕТОД ПРОГНОЗУ ТА КОРЕКЦІЇ АДАМСА 2-ГО ПОРЯДКУ
# =====================================================================
def adams_pc_2_fixed(x0, y0, x_max, h):
    # Метод потребує 2 початкові точки. Другу знайдемо через РК-4.
    x_vals = [x0, x0 + h]
    y_vals = [y0, runge_kutta_4_step(x0, y0, h)]
    
    x = x0 + h
    while x < x_max - 1e-10:
        if x + h > x_max:
            h = x_max - x
            
        x_next = x + h
        # Етап прогнозу (явний)
        f_n = f(x_vals[-1], y_vals[-1])
        f_prev = f(x_vals[-2], y_vals[-2])
        y_pred = y_vals[-1] + (h / 2) * (3 * f_n - f_prev)
        
        # Етап корекції (ітераційний процесор з критерієм зупинки)
        y_corr = y_pred
        eps_iter = 1e-8
        for i in range(10):
            y_corr_next = y_vals[-1] + (h / 2) * (f(x_next, y_corr) + f_n)
            if abs(y_corr_next - y_corr) <= eps_iter:
                y_corr = y_corr_next
                break
            y_corr = y_corr_next
            
        y_vals.append(y_corr)
        x_vals.append(x_next)
        x = x_next
        
    return np.array(x_vals), np.array(y_vals)

def adams_pc_2_adaptive(x0, y0, x_max, eps):
    x_vals = [x0]
    y_vals = [y0]
    h_vals = []
    
    x = x0
    h = 0.05
    
    # Знаходимо другу точку через РК-4 для старту
    y_next = runge_kutta_4_step(x, y0, h)
    x_vals.append(x + h)
    y_vals.append(y_next)
    h_vals.append(h)
    x += h
    
    while x < x_max - 1e-10:
        if x + h > x_max:
            h = x_max - x
            
        x_next = x + h
        f_n = f(x_vals[-1], y_vals[-1])
        f_prev = f(x_vals[-2], y_vals[-2])
        
        # Прогноз
        y_pred = y_vals[-1] + (h / 2) * (3 * f_n - f_prev)
        # Модифікація та корекція
        y_corr = y_vals[-1] + (h / 2) * (f(x_next, y_pred) + f_n)
        
        # Оцінка локальної похибки через (y_corr - y_pred)
        error = (1.0 / 6.0) * abs(y_corr - y_pred)
        
        if error > eps:
            h /= 2
            # Переобчислюємо проміжний вузол через РК-4
            x_vals = x_vals[:-1]
            y_vals = y_vals[:-1]
            x = x_vals[-1]
            y_next = runge_kutta_4_step(x, y_vals[-1], h)
            x_vals.append(x + h)
            y_vals.append(y_next)
            x += h
            continue
        else:
            h_vals.append(h)
            y_vals.append(y_corr)
            x_vals.append(x_next)
            x = x_next
            if error < eps / 8:
                h *= 2
                
    return np.array(x_vals), np.array(y_vals), h_vals

# =====================================================================
# ОБЧИСЛЕННЯ ТА ПОБУДОВА ГРАФІКІВ
# =====================================================================
x0, y0, x_max = 0.0, 1.0, 1.5
h_fixed = 0.01
eps_target = 1e-5

# 1. Фіксований крок РК-4 та Адамс
x_rk, y_rk = runge_kutta_4_fixed(x0, y0, x_max, h_fixed)
x_ad, y_ad = adams_pc_2_fixed(x0, y0, x_max, h_fixed)

# 2. Адаптивний крок
x_rk_ad, y_rk_ad, h_rk_ad = runge_kutta_4_adaptive(x0, y0, x_max, eps_target)
x_ad_ad, y_ad_ad, h_ad_ad = adams_pc_2_adaptive(x0, y0, x_max, eps_target)

# Візуалізація результатів
plt.figure(figsize=(12, 10))

# Графік 1: Локальна похибка методів з фіксованим кроком (Пункт 3, 7)
plt.subplot(2, 2, 1)
plt.plot(x_rk, y_rk - y_exact(x_rk), 'b-', label='Похибка РК-4')
plt.plot(x_ad, y_ad - y_exact(x_ad), 'r--', label='Похибка Адамса (ПК-2)')
plt.title("Справжня локальна похибка $\\varphi_n = y_n - y(x_n)$")
plt.xlabel("x")
plt.ylabel("Похибка")
plt.grid(True)
plt.legend()

# Графік 2: Оцінка похибки Адамса через предикатор-коректор (Пункт 4)
plt.subplot(2, 2, 2)
f_n_vec = np.array([f(xi, yi) for xi, yi in zip(x_ad[:-1], y_ad[:-1])])
f_prev_vec = np.array([f(xi, yi) for xi, yi in zip(x_ad[:-2], y_ad[:-2])])
# Обчислимо y_pred для фіксованого масиву
y_pred_vec = y_ad[1:-1] + (h_fixed / 2) * (3 * f_n_vec[1:] - f_prev_vec)
est_err_ad = (1.0 / 6.0) * abs(y_ad[2:] - y_pred_vec)
plt.plot(x_ad[2:], est_err_ad, 'g-', label='Оцінка $(y^{kop} - y^{np})/6$')
plt.title("Оцінка похибки методом предикатор-коректор")
plt.xlabel("x")
plt.ylabel("Оцінка похибки")
plt.grid(True)
plt.legend()

# Графік 3: Автоматичний вибір кроку для Адамса (Пункт 5)
plt.subplot(2, 2, 3)
plt.step(x_ad_ad[:-1], h_ad_ad, 'r-', where='post', label='Крок $h(x)$ для Адамса')
plt.title(f"Адаптивний крок Адамса ($\\epsilon = {eps_target}$)")
plt.xlabel("x")
plt.ylabel("Величина кроку h")
plt.grid(True)
plt.legend()

# Графік 4: Автоматичний вибір кроку для Рунге-Кутта (Пункт 9)
plt.subplot(2, 2, 4)
plt.step(x_rk_ad[:-1], h_rk_ad, 'b-', where='post', label='Крок $h(x)$ для РК-4')
plt.title(f"Адаптивний крок Рунге-Кутта ($\\epsilon = {eps_target}$)")
plt.xlabel("x")
plt.ylabel("Величина кроку h")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()