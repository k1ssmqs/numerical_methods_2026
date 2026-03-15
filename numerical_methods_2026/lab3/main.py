import numpy as np
import matplotlib.pyplot as plt
import csv

# Зчитування даних з CSV
monthss = []
temps = []

with open("data.csv", newline="") as file:
    reader = csv.DictReader(file)
    for row in reader:
        monthss.append(float(row["Month"]))
        temps.append(float(row["Temp"]))

months = np.array(months)
temps = np.array(temps)


# Формування матриці для МНК
def build_matrix(x_vals, degree):
    size = degree + 1
    mat = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            mat[i, j] = np.sum(x_vals ** (i + j))
    return mat


# Формування вектора правої частини для МНК
def build_vector(x_vals, y_vals, degree):
    size = degree + 1
    vec = np.zeros(size)
    for i in range(size):
        vec[i] = np.sum(y_vals * (x_vals ** i))
    return vec


# Розв'язок СЛАР методом Гауса
def solve_gauss(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)

    for k in range(n):
        pivot_row = np.argmax(np.abs(A[k:, k])) + k
        A[[k, pivot_row]] = A[[pivot_row, k]]
        b[[k, pivot_row]] = b[[pivot_row, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = np.sum(A[i, i + 1:] * x_sol[i + 1:])
        x_sol[i] = (b[i] - s) / A[i, i]
    return x_sol


# Обчислення значень полінома
def eval_poly(x_vals, coefs):
    y_vals = np.zeros_like(x_vals, dtype=float)
    for i, c in enumerate(coefs):
        y_vals += c * (x_vals **  i)
    return y_vals


# Дисперсія апроксимації
def calc_variance(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Пошук оптимального степеня полінома
max_deg = 4
var_list = []

for deg in range(1, max_deg + 1):
    mat = build_matrix(months, deg)
    vec = build_vector(months, temps, deg)
    coeffs = solve_gauss(mat, vec)
    y_fit = eval_poly(months, coeffs)
    var_list.append(calc_variance(temps, y_fit))

best_deg = np.argmin(var_list) + 1

# Побудова апроксимації для оптимального степеня
mat = build_matrix(months, best_deg)
vec = build_vector(months, temps, best_deg)
coeffs = solve_gauss(mat, vec)
y_fit = eval_poly(months, coeffs)

# Прогноз на майбутні місяці
future_months = np.array([25, 26, 27])
future_temps = eval_poly(future_months, coeffs)

# Похибка апроксимації
errors = temps - y_fit

# Вивід результатів
print("Дисперсії за степенями:")
for i, v in enumerate(var_list, start=1):
    print(f"Степінь {i}: {v:.4f}")
print("\nОптимальний степінь:", best_deg)

print("\nКоефіцієнти полінома:")
for i, c in enumerate(coeffs):
    print(f"a{i} = {c:.5f}")

print("\nПрогноз температур:")
for m, t in zip(future_months, future_temps):
    print(f"Місяць {int(m)} -> {t:.2f} °C")

# Графік апроксимації та фактичних даних
plt.figure(figsize=(10, 6))
plt.plot(months, temps, 'o', color='purple', markersize=5, label="Фактичні дані")
plt.plot(months, y_fit, '-', color='blue', linewidth=2, label="Апроксимація")
plt.xlabel("Місяць")
plt.ylabel("Температура")
plt.title("Апроксимація температури")
plt.legend()
plt.grid(True)
plt.show()

# Графік похибки
abs_error = np.abs(errors)
plt.figure(figsize=(10, 6))
plt.bar(months, abs_error, alpha=0.6, color='yellow', label="Абсолютна похибка")
plt.plot(months, abs_error, marker='o', color='orange', linewidth=2, label="Лінія похибки")
plt.title("Графік похибки апроксимації")
plt.xlabel("Місяць")
plt.ylabel("|Temp_fact - Temp_approx|")
plt.grid(True)
plt.legend()
plt.show()