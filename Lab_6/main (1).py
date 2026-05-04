import random

def generate_and_write_data(n, exact_x):
    with open("A.txt", "w") as file_a, open("B.txt", "w") as file_b:
        A = []
        for i in range(n):
            row = []
            sum_for_b = 0
            for j in range(n):
                #генеруємо випадкове число від -10 до 10
                val = random.uniform(-10.0, 10.0)
                #додаємо діагональне переважання для стабільності LU-розкладу
                if i == j:
                    val += 100.0 if val > 0 else -100.0
                
                row.append(val)
                file_a.write(f"{val:.6f} ")
                sum_for_b += val * exact_x # b_i = sum(a_ij * x_j)
                
            A.append(row)
            file_a.write("\n")
            file_b.write(f"{sum_for_b:.6f}\n")
            
    print("Файли A.txt та B.txt згенеровано.")

#зчитування матриці А з текстового файлу
def read_matrix(filename):
    A = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip(): # перевірка на порожній рядок
                row = [float(x) for x in line.split()]
                A.append(row)
    return A

#зчитування вектора В з текстового файлу
def read_vector(filename):
    B = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                B.append(float(line.strip()))
    return B

#знаходження LU-розкладу матриці А
def find_lu(A, n):
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    
    #одиниці на головній діагоналі матриці U
    for i in range(n):
        U[i][i] = 1.0

    for k in range(n):
        # k-й стовпець матриці L
        for i in range(k, n):
            sum_l = sum(L[i][j] * U[j][k] for j in range(k))
            L[i][k] = A[i][k] - sum_l
            
        # k-й рядок матриці U
        for i in range(k + 1, n):
            sum_u = sum(L[k][j] * U[j][i] for j in range(k))
            U[k][i] = (A[k][i] - sum_u) / L[k][k]
            
    return L, U

#запис LU-розкладу матриці А в текстовий файл
def write_lu_to_file(L, U, n):
    with open("LU.txt", "w") as f:
        for i in range(n):
            for j in range(n):
                if i >= j:
                    f.write(f"{L[i][j]:.6f} ")
                else:
                    f.write(f"{U[i][j]:.6f} ")
            f.write("\n")
    print("LU-розклад записано у файл LU.txt.")

#розв'язок системи рівнянь AX=B за допомогою LU-розкладу
def solve_system(L, U, B, n):
    Z = [0.0] * n
    X = [0.0] * n

    #прямий хід
    for i in range(n):
        s = sum(L[i][j] * Z[j] for j in range(i))
        Z[i] = (B[i] - s) / L[i][i]

    #зворотній хід
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * X[j] for j in range(i + 1, n))
        X[i] = Z[i] - s

    return X

# обчислення добутку матриці на вектор
def multiply_mat_vec(A, X, n):
    res = [0.0] * n
    for i in range(n):
        res[i] = sum(A[i][j] * X[j] for j in range(n))
    return res

# максимальний за модулем елемент вектора V
def calculate_vector_norm(V):
    return max(abs(v) for v in V)

def main():
    n = 100
    exact_x = 2.5
    
    #генеруємо та записуємо A і B
    generate_and_write_data(n, exact_x)
    
    #зчитуємо A та B з файлів
    A = read_matrix("A.txt")
    B = read_vector("B.txt")
    
    #знаходимо та зберігаємо LU-розклад
    L, U = find_lu(A, n)
    write_lu_to_file(L, U, n)
    
    #розв'язуємо систему
    X = solve_system(L, U, B, n)
    
    #похибка eps
    AX = multiply_mat_vec(A, X, n)
    R = [AX[i] - B[i] for i in range(n)]
    
    initial_eps = calculate_vector_norm(R)
    print(f"Точність початкового розв'язку (eps): {initial_eps:e}")

    # 5. Ітераційний метод уточнення розв'язку
    eps0 = 1e-14
    iterations = 0
    dX = [0.0] * n
    
    while True:
        # R = B - AX (вектор нев'язки)
        AX = multiply_mat_vec(A, X, n)
        R = [B[i] - AX[i] for i in range(n)]
        
        # A * dX = R  =>  розв'язуємо через готовий LU
        dX = solve_system(L, U, R, n)
        
        # X = X + dX
        for i in range(n):
            X[i] += dX[i]
            
        iterations += 1
        
        current_eps = calculate_vector_norm(dX)
        if current_eps <= eps0: # Умова виходу
            break
            
        if iterations > 100:
            print("Досягнуто ліміт ітерацій!")
            break
            
    print(f"Кількість ітерацій уточнення: {iterations}")
    print(f"Кінцева похибка (norm dX): {current_eps:e}")
    print(f"Перевірка: X[0] = {X[0]:.5f}, X[{n-1}] = {X[n-1]:.5f} (очікується 2.5)")

if __name__ == "__main__":
    main()