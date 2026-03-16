import numpy as np
import matplotlib.pyplot as plt

rps_values = [50, 100, 200, 400, 800]
cpu_values = [20, 35, 60, 110, 210]


def build_difference_table(x_vals, y_vals):
    size = len(y_vals)
    table = np.zeros((size, size))
    table[:, 0] = y_vals

    for col in range(1, size):
        for row in range(size - col):
            table[row][col] = (table[row + 1][col - 1] - table[row][col - 1]) / (x_vals[row + col] - x_vals[row])

    return table


def evaluate_newton(x_vals, diff_table, x_point):
    degree = len(x_vals) - 1
    result = diff_table[0][degree]

    for i in range(1, degree + 1):
        result = diff_table[0][degree - i] + (x_point - x_vals[degree - i]) * result

    return result


difference_table = build_difference_table(rps_values, cpu_values)

prediction = evaluate_newton(rps_values, difference_table, 600)

print("CPU при 600 RPS =", prediction)

x_graph = np.linspace(50, 800, 100)
y_graph = [evaluate_newton(rps_values, difference_table, value) for value in x_graph]

plt.scatter(rps_values, cpu_values, color="green")
plt.plot(x_graph, y_graph, color="green")

plt.xlabel("RPS")
plt.ylabel("CPU")
plt.grid()
plt.show()

print("Таблиця розділених різниць:")
print(difference_table)
prediction = evaluate_newton(rps_values, difference_table, 600)
print("CPU при 600 RPS =", prediction)

# Дослідження з меншою кількістю вузлів
rps_small = rps_values[:3]
cpu_small = cpu_values[:3]
table_small = build_difference_table(rps_small, cpu_small)
prediction_small = evaluate_newton(rps_small, table_small, 600)
print("CPU (3 вузли) =", prediction_small)
rps_medium = rps_values[:4]
cpu_medium = cpu_values[:4]
table_medium = build_difference_table(rps_medium, cpu_medium)
prediction_medium = evaluate_newton(rps_medium, table_medium, 600)
print("CPU (4 вузли) =", prediction_medium)