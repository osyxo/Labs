import math
import numpy as np

def f(x):
    """Задана трансцендентна функція f(x) = x*cos(x) - sin(x)"""
    return x * math.cos(x) - math.sin(x)

def f_prime(x):
    """Похідна f'(x) = cos(x) - x*sin(x) - cos(x) = -x*sin(x)"""
    return -x * math.sin(x)

def f_double_prime(x):
    """Друга похідна f''(x) = -sin(x) - x*cos(x)"""
    return -math.sin(x) - x * math.cos(x)

# ПУНКТ 1. Табуляція функції та знаходження початкових наближень коренів
def tabulate_and_find_roots(a, b, h, filename="tabulation.txt"):
    """
    Табуляція функції f(x) на відрізку [a, b] з кроком h.
    Запис результатів у текстовий файл.
    Знаходження наближених абсцис перетину з віссю OX (зміна знаку).
    Повертає два початкові наближення: одне при зростанні, одне при спаданні.
    """
    x_vals = []
    f_vals = []
    x = a
    while x <= b + 1e-9:
        x_vals.append(x)
        f_vals.append(f(x))
        x = round(x + h, 10)

    # Запис у файл
    with open(filename, "w") as fout:
        fout.write(f"{'x':>10}  {'f(x)':>15}\n")
        fout.write("-" * 28 + "\n")
        for xi, fi in zip(x_vals, f_vals):
            fout.write(f"{xi:10.4f}  {fi:15.8f}\n")

    # Знаходження відрізків, де функція змінює знак
    sign_changes = []
    for i in range(len(x_vals) - 1):
        if x_vals[i] != 0.0 and f_vals[i] * f_vals[i + 1] < 0:
            # Зберігаємо середину відрізку як початкове наближення
            x_mid = (x_vals[i] + x_vals[i + 1]) / 2
            # Визначаємо: зростання чи спадання функції в точці перетину
            behavior = "зростає" if f_vals[i + 1] > f_vals[i] else "спадає"
            sign_changes.append((x_mid, behavior, x_vals[i], x_vals[i + 1]))

    print(f"Пункт 1. Табуляція f(x) записана у '{filename}'")
    print(f"Знайдено {len(sign_changes)} відрізків зміни знаку:")
    for sc in sign_changes:
        print(f"  x ∈ [{sc[2]:.2f}, {sc[3]:.2f}], x0 ≈ {sc[0]:.4f}, функція {sc[1]}")

    # Вибираємо два корені з різною поведінкою (зростання і спадання)
    rising = next((sc for sc in sign_changes if sc[1] == "зростає"), None)
    falling = next((sc for sc in sign_changes if sc[1] == "спадає"), None)

    x0_rise = rising[0] if rising else sign_changes[0][0]
    x0_fall = falling[0] if falling else sign_changes[1][0]

    print(f"\nВибрано два початкових наближення:")
    print(f"  x0_1 = {x0_rise:.4f}  (функція зростає)")
    print(f"  x0_2 = {x0_fall:.4f}  (функція спадає)")
    return x0_rise, x0_fall

# =============================================================================
# ПУНКТ 2. Методи розв'язку нелінійного рівняння
# ПУНКТ 3. Критерій зупинки: |x_{k+1} - x_k| < eps  і  |f(x_{k+1})| < eps
# =============================================================================

def stop_criterion(x_new, x_old, eps):
    """Пункт 3: одночасне виконання двох умов зупинки"""
    return abs(x_new - x_old) < eps and abs(f(x_new)) < eps

# --- Метод простої ітерації (релаксації) ---
def simple_iteration(x0, eps=1e-10, max_iter=10000):
    """
    Пункт 2: Метод простої ітерації (релаксації).
    x_{k+1} = x_k - tau * f(x_k)
    tau вибирається як 1 / max|f'(x)| на відрізку.
    """
    # Оцінка tau: беремо tau = 1 / max|f'(x0)|, щоб |1 - tau*f'(x)| < 1
    fp = f_prime(x0)
    if abs(fp) < 1e-14:
        tau = 0.1
    else:
        tau = 1.0 / abs(fp)

    x = x0
    for k in range(max_iter):
        x_new = x - tau * f(x)
        if stop_criterion(x_new, x, eps):
            return x_new, k + 1
        x = x_new
    return x, max_iter

# --- Метод Ньютона ---
def newton(x0, eps=1e-10, max_iter=1000):
    """
    Пункт 2: Метод Ньютона (другого порядку збіжності).
    x_{k+1} = x_k - f(x_k) / f'(x_k)
    """
    x = x0
    for k in range(max_iter):
        fp = f_prime(x)
        if abs(fp) < 1e-14:
            print("  [Ньютон] f'(x) ≈ 0, зупинка")
            return x, k
        x_new = x - f(x) / fp
        if stop_criterion(x_new, x, eps):
            return x_new, k + 1
        x = x_new
    return x, max_iter

# --- Метод Чебишева ---
def chebyshev(x0, eps=1e-10, max_iter=1000):
    """
    Пункт 2: Метод Чебишева (порядок збіжності ~3).
    x_{k+1} = x_k - f/f' - (f^2 * f'') / (2 * (f')^3)
    """
    x = x0
    for k in range(max_iter):
        fv = f(x)
        fp = f_prime(x)
        fpp = f_double_prime(x)
        if abs(fp) < 1e-14:
            return x, k
        x_new = x - fv / fp - (fv**2 * fpp) / (2 * fp**3)
        if stop_criterion(x_new, x, eps):
            return x_new, k + 1
        x = x_new
    return x, max_iter

# --- Метод хорд ---
def chord_method(x0, eps=1e-10, max_iter=1000):
    """
    Пункт 2: Метод хорд (багатокроковий).
    x_{k+1} = x_k - f(x_k) * (x_k - x_{k-1}) / (f(x_k) - f(x_{k-1}))
    Потребує двох початкових наближень x0 і x1 = x0 + small_step.
    """
    x_prev = x0
    x_curr = x0 + 0.1 * (1 if f_prime(x0) > 0 else -1)
    for k in range(max_iter):
        fv_curr = f(x_curr)
        fv_prev = f(x_prev)
        denom = fv_curr - fv_prev
        if abs(denom) < 1e-14:
            return x_curr, k
        x_new = x_curr - fv_curr * (x_curr - x_prev) / denom
        if stop_criterion(x_new, x_curr, eps):
            return x_new, k + 1
        x_prev = x_curr
        x_curr = x_new
    return x_curr, max_iter

# --- Метод парабол ---
def parabola_method(x0, eps=1e-10, max_iter=1000):
    """
    Пункт 2: Метод парабол (метод Мюллера, порядок збіжності ~1.84).
    Будується інтерполяційна парабола через три вузли;
    нове наближення — найближчий нуль параболи.
    """
    # Три початкових вузли поблизу x0
    x0_ = x0 - 0.2
    x1_ = x0 - 0.1
    x2_ = x0

    for k in range(max_iter):
        try:
            f0, f1, f2 = f(x0_), f(x1_), f(x2_)
        except (ValueError, OverflowError):
            break

        h1 = x1_ - x0_
        h2 = x2_ - x1_
        d1 = (f1 - f0) / h1 if abs(h1) > 1e-14 else 0.0
        d2 = (f2 - f1) / h2 if abs(h2) > 1e-14 else 0.0
        a  = (d2 - d1) / (h1 + h2) if abs(h1 + h2) > 1e-14 else 0.0
        b  = d2 + a * h2
        c  = f2

        disc = b * b - 4 * a * c
        if disc < 0:
            disc = 0.0
        denom = b + math.sqrt(disc) if abs(b + math.sqrt(disc)) >= abs(b - math.sqrt(disc)) else b - math.sqrt(disc)
        if abs(denom) < 1e-14:
            break
        dx = -2 * c / denom
        x_new = x2_ + dx

        if abs(x_new) > 1e6:
            break
        if stop_criterion(x_new, x2_, eps):
            return x_new, k + 1
        x0_, x1_, x2_ = x1_, x2_, x_new

    return x2_, max_iter

# --- Метод зворотної інтерполяції (3 вузли) ---
def inverse_interpolation(x0, eps=1e-10, max_iter=1000):
    """
    Пункт 2: Метод зворотної інтерполяції.
    Будується інтерполяційний многочлен Лагранжа L(y) на трьох вузлах
    (f(x_k-2), f(x_k-1), f(x_k)) і знаходиться x = L(0).
    """
    x_prev2 = x0 - 0.2
    x_prev1 = x0 - 0.1
    x_curr  = x0
    for k in range(max_iter):
        y0, y1, y2 = f(x_prev2), f(x_prev1), f(x_curr)
        # Формула Лагранжа при y=0
        denom0 = (y0 - y1) * (y0 - y2)
        denom1 = (y1 - y0) * (y1 - y2)
        denom2 = (y2 - y0) * (y2 - y1)
        if abs(denom0) < 1e-14 or abs(denom1) < 1e-14 or abs(denom2) < 1e-14:
            # Не можемо продовжити — переходимо до методу Ньютона
            return newton(x_curr, eps, max_iter - k)
        x_new = (x_prev2 * (-y1) * (-y2) / denom0
               + x_prev1 * (-y0) * (-y2) / denom1
               + x_curr  * (-y0) * (-y1) / denom2)
        if stop_criterion(x_new, x_curr, eps):
            return x_new, k + 1
        x_prev2 = x_prev1
        x_prev1 = x_curr
        x_curr  = x_new
    return x_curr, max_iter

# =============================================================================
# ПУНКТ 4. Порівняння кількості ітерацій для двох коренів та всіх методів
# =============================================================================
def compare_methods(x0_rise, x0_fall, eps=1e-10):
    """Пункт 4: запуск усіх методів для обох початкових наближень."""
    methods = [
        ("Проста ітерація",        simple_iteration),
        ("Ньютон",                  newton),
        ("Чебишев",                 chebyshev),
        ("Хорд",                    chord_method),
        ("Парабол",                 parabola_method),
        ("Зворотна інтерполяція",   inverse_interpolation),
    ]
    print(f"\nПункт 4. Кількість ітерацій (eps = {eps}):")
    print(f"{'Метод':<26} | {'Корінь 1 (зростає)':<22} | {'Корінь 2 (спадає)':<22}")
    print("-" * 76)
    for name, method in methods:
        x1, n1 = method(x0_rise, eps)
        x2, n2 = method(x0_fall, eps)
        print(f"{name:<26} | x={x1:9.6f}, iter={n1:<5} | x={x2:9.6f}, iter={n2:<5}")

# =============================================================================
# ПУНКТИ 5–6. Алгебраїчне рівняння 3-го порядку з коефіцієнтами у файлі
# p(x) = x^3 - 3x^2 + 4  (один дійсний корінь ≈ -0.828, два комплексних)
# =============================================================================
POLY_COEFFS = [1, -3, 0, 4]   # коефіцієнти від x^3 до x^0

def write_poly_coefficients(coeffs, filename="poly_coeffs.txt"):
    """Пункт 6: записати коефіцієнти алгебраїчного рівняння у текстовий файл."""
    with open(filename, "w") as fout:
        fout.write("# Коефіцієнти алгебраїчного рівняння від старшого до вільного члена\n")
        fout.write(" ".join(map(str, coeffs)) + "\n")
    print(f"\nПункт 6. Коефіцієнти рівняння записані у '{filename}'")

# =============================================================================
# ПУНКТ 7. Зчитування коефіцієнтів та обчислення значення многочлена
# =============================================================================
def read_poly_coefficients(filename="poly_coeffs.txt"):
    """Пункт 7: зчитати коефіцієнти довільного алгебраїчного многочлена з файлу."""
    with open(filename, "r") as fin:
        for line in fin:
            line = line.strip()
            if line and not line.startswith("#"):
                return list(map(float, line.split()))
    return []

def poly_value(coeffs, x):
    """Пункт 7: обчислення значення многочлена для заданого x."""
    result = 0.0
    for c in coeffs:
        result = result * x + c
    return result

# =============================================================================
# ПУНКТ 8. Метод Ньютона зі схемою Горнера для дійсних коренів
# =============================================================================
def horner(coeffs, x):
    """
    Схема Горнера: повертає (p(x), p'(x)) одночасно.
    b_n = a_n,  b_{k-1} = a_{k-1} + b_k * x
    c_n = b_n,  c_{k-1} = b_{k-1} + c_k * x  (де c — для похідної)
    """
    n = len(coeffs) - 1
    b = coeffs[0]
    c = 0.0
    for i in range(1, n + 1):
        c = b + c * x
        b = coeffs[i] + b * x
    # b = p(x), c = p'(x)
    return b, c

def newton_horner(coeffs, x0, eps=1e-10, max_iter=1000):
    """
    Пункт 8: Метод Ньютона зі схемою Горнера для знаходження дійсного кореня.
    """
    x = x0
    for k in range(max_iter):
        px, dpx = horner(coeffs, x)
        if abs(dpx) < 1e-14:
            break
        x_new = x - px / dpx
        if abs(x_new - x) < eps and abs(px) < eps:
            return x_new, k + 1
        x = x_new
    return x, max_iter

# =============================================================================
# ПУНКТ 9. Метод Ліна для знаходження комплексних коренів
# =============================================================================
def lin_method(coeffs, p0, q0, eps=1e-8, max_iter=1000):
    """
    Пункт 9: Метод Ліна для знаходження пари комплексно-спряжених коренів.
    Шукаємо множник (x^2 + p*x + q) алгебраїчного рівняння.
    Ділимо многочлен на (x^2 + p*x + q) з остачею r1*x + r0,
    і ітеративно уточнюємо p, q поки залишок → 0.
    """
    n = len(coeffs) - 1
    p = p0
    q = q0

    for iteration in range(max_iter):
        # Ділення coeffs на (x^2 + p*x + q)
        # b[i] — частка, остача: b[n-1]*x + b[n]
        b = [0.0] * (n + 1)
        b[0] = coeffs[0]
        if n >= 1:
            b[1] = coeffs[1] - p * b[0]
        for i in range(2, n + 1):
            b[i] = coeffs[i] - p * b[i-1] - q * b[i-2]

        r1 = b[n-1]   # коефіцієнт при x в остачі
        r0 = b[n]     # вільний член остачі

        if abs(r0) < eps and abs(r1) < eps:
            break

        # Ділення b[0..n-2] на (x^2 + p*x + q) → c, остача c[n-2]*x + c[n-1]
        m = n - 1  # степінь b[0..n-2]
        c = [0.0] * (m + 1)
        c[0] = b[0]
        if m >= 1:
            c[1] = b[1] - p * c[0]
        for i in range(2, m + 1):
            c[i] = b[i] - p * c[i-1] - q * c[i-2]

        # Елементи матриці 2×2 для системи Δp, Δq
        # ∂r1/∂p ≈ -c[n-2], ∂r1/∂q ≈ -c[n-3]
        # ∂r0/∂p ≈ -c[n-2]*... → використовуємо спрощену версію:
        # J * [Δp, Δq]^T = -[r1, r0]^T
        # J = [[-c[m-1], -c[m-2]], [-c[m-2], -c[m-3] + ...]]
        # Для ступеня 3 (n=3, m=2):
        # c = [c0, c1, c2];  r1=b[2], r0=b[3]
        # Δp, Δq з системи:
        # c[1]*Δp + c[0]*Δq = r1
        # c[0]*Δp             = r0  (спрощено для n=3)
        if n == 3:
            # Для n=3: b=[b0,b1,b2,b3], c=[b0] (частка від ділення b[0..1] на квадрат)
            # Система: b0*Δp = r0;  b1*Δp + b0*Δq = r1  → підставляємо
            if abs(b[0]) < 1e-14:
                break
            dp = r0 / b[0]
            dq = (r1 - b[1] * dp) / b[0]
        else:
            # Загальний випадок через матрицю Якобі
            J11 = -c[m-1] if m >= 1 else 0.0
            J12 = -c[m-2] if m >= 2 else 0.0
            J21 = -c[m-2] if m >= 2 else 0.0
            J22 = -(c[m-3] if m >= 3 else 0.0)
            det = J11 * J22 - J12 * J21
            if abs(det) < 1e-14:
                break
            dp = (-r1 * J22 + r0 * J12) / det
            dq = (-r0 * J11 + r1 * J21) / det

        p += dp
        q += dq

    # Корені x^2 + p*x + q = 0
    discriminant = p * p - 4 * q
    if discriminant >= 0:
        z1 = (-p + math.sqrt(discriminant)) / 2
        z2 = (-p - math.sqrt(discriminant)) / 2
    else:
        re_part = -p / 2
        im_part = math.sqrt(-discriminant) / 2
        z1 = complex(re_part,  im_part)
        z2 = complex(re_part, -im_part)

    return z1, z2, iteration + 1


# =============================================================================
# ГОЛОВНА ПРОГРАМА
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  Лабораторна робота №8: Чисельні методи розв'язку нелінійних рівнянь")
    print("=" * 70)

    # --- Пункт 1 ---
    x0_rise, x0_fall = tabulate_and_find_roots(-10, 10, 0.1, "tabulation.txt")

    # --- Пункти 2–4 ---
    EPS = 1e-10
    compare_methods(x0_rise, x0_fall, EPS)

    # --- Пункти 5–6 ---
    print("\nПункт 5. Алгебраїчне рівняння: p(x) = x^3 - 3x^2 + 4")
    print("  Коефіцієнти: a3=1, a2=-3, a1=0, a0=4")
    write_poly_coefficients(POLY_COEFFS, "poly_coeffs.txt")

    # --- Пункт 7 ---
    coeffs = read_poly_coefficients("poly_coeffs.txt")
    print(f"\nПункт 7. Зчитані коефіцієнти: {coeffs}")
    test_x = 2.0
    print(f"  p({test_x}) = {poly_value(coeffs, test_x):.6f}")

    # --- Пункт 8 ---
    # p(x)=x^3-3x^2+4 має дійсні корені: x=-1, x=2 (double).
    # Беремо x0=-0.5 → збіжиться до x=-1
    real_root_x0 = -0.5
    real_root, iters = newton_horner(coeffs, real_root_x0, eps=1e-10)
    print(f"\nПункт 8. Метод Ньютона (схема Горнера):")
    print(f"  Дійсний корінь: x ≈ {real_root:.8f}, кількість ітерацій: {iters}")
    print(f"  Перевірка: p({real_root:.8f}) = {poly_value(coeffs, real_root):.2e}")

    # --- Пункт 9 ---
    # Метод Ліна шукає квадратний множник (x^2+px+q) рівняння.
    # Для рівняння q(x) = x^3 - x^2 + x - 1 = (x-1)(x^2+1)
    # корені: x=1, z=+i, z=-i; квадратний множник: x^2+1 (p=0, q=1).
    # Метод збігається лише при достатньо близькому початковому наближенні.
    poly_complex = [1, -1, 1, -1]
    print(f"\nПункт 9. Метод Ліна для q(x) = x³ - x² + x - 1")
    print(f"  (корені: x=1, x=+i, x=-i)")
    # Початкове наближення: p0=0.01, q0=0.99 (близько до p=0, q=1)
    z1, z2, lin_iters = lin_method(poly_complex, p0=0.01, q0=0.99, eps=1e-8)
    print(f"  Метод Ліна (комплексні корені):")
    print(f"  z1 = {z1},  z2 = {z2}")
    print(f"  Кількість ітерацій: {lin_iters}")
    if isinstance(z1, complex):
        val = poly_value(poly_complex, z1)
        print(f"  Перевірка |q(z1)| = {abs(val):.2e}")
    import numpy as np
    np_roots = np.roots(poly_complex)
    print(f"  Контроль (numpy): {np_roots}")

    print("\nГотово.")