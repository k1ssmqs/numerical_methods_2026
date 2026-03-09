import requests
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

response = requests.get(url)
data = response.json()
results = data["results"]
n_nodes = len(results)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**  2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]

distances = [0]
for idx in range(1, n_nodes):
    dist = haversine(*coords[idx - 1], *coords[idx])
    distances.append(distances[-1] + dist)

with open("tabulation.txt", "w") as f:
    f.write(" Latitude | Longitude | Elevation (m) | Distance (m)\n")
    for idx, point in enumerate(results):
        f.write(
            f"{idx:2d} | {point['latitude']:.6f} |"
            f" {point['longitude']:.6f} |"
            f" {point['elevation']:.2f} |"
            f" {distances[idx]:.2f}\n")


x_full = np.array(distances)
y_full = np.array(elevations)


def cubic_spline_natural(x_arr, y_arr):
    num = len(x_arr)
    h = np.diff(x_arr)

    A = np.zeros(num)
    B = np.zeros(num)
    C = np.zeros(num)
    D = np.zeros(num)

    B[0] = 1
    B[-1] = 1

    for i in range(1, num - 1):
        A[i] = h[i - 1]
        B[i] = 2 * (h[i - 1] + h[i])
        C[i] = h[i]
        D[i] = 6 * ((y_arr[i + 1] - y_arr[i])/
                    h[i] - (y_arr[i] - y_arr[i - 1])/
                    h[i - 1])

    for i in range(1, num):
        m_val = A[i] / B[i - 1]
        B[i] -= m_val * C[i - 1]
        D[i] -= m_val * D[i - 1]

    M = np.zeros(num)
    M[-1] = D[-1] / B[-1]
    for i in range(num - 2, -1, -1):
        M[i] = (D[i] - C[i] * M[i + 1]) / B[i]

    a = y_arr[:-1]
    b = np.zeros(num - 1)
    c = M[:-1] / 2
    d = np.zeros(num - 1)

    for i in range(num - 1):
        b[i] = ((y_arr[i + 1] - y_arr[i])/
                h[i] - h[i] * (2 * M[i] + M[i + 1]) /6)

        d[i] = (M[i + 1] - M[i]) / (6 * h[i])

    return a, b, c, d, x_arr


def spline_eval(xi, a, b, c, d, x_nodes):
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= xi <= x_nodes[i + 1]:
            dx = xi - x_nodes[i]
            return a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx  **3
    return None


a_full, b_full, c_full, d_full, x_nodes_full = cubic_spline_natural(x_full, y_full)

xx = np.linspace(x_full[0], x_full[-1], 1000)
yy_full = np.array([spline_eval(xi, a_full, b_full, c_full, d_full, x_nodes_full) for xi in xx])


def test_nodes(k):
    indices = np.linspace(0, len(x_full) - 1, k, dtype=int)
    x_k = x_full[indices]
    y_k = y_full[indices]

    a_k, b_k, c_k, d_k, x_nodes_k = cubic_spline_natural(x_k, y_k)
    yy_k = np.array([spline_eval(xi, a_k, b_k, c_k, d_k, x_nodes_k) for xi in xx])

    error = np.abs(yy_k - yy_full)
    print(f"--- {k} вузлів --- Макс. похибка: {np.max(error):.2f} м")
    return yy_k, error


yy_10, err_10 = test_nodes(10)
yy_15, err_15 = test_nodes(15)
yy_20, err_20 = test_nodes(20)

total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n_nodes))
grad_full = np.gradient(yy_full, xx) * 100
mass = 80
g = 9.81
energy = mass * g * total_ascent

print(f"Загальна відстань: {distances[-1]:.0f} м")
print(f"Сумарний підйом: {total_ascent:.0f} м")
print(f"Максимальний градієнт: {np.max(grad_full):.1f}%")
print(f"Енергія на підйом (80 кг): {energy / 4184:.0f} ккал")
plt.figure(figsize=(11, 6))
plt.plot(distances, elevations, 'o-', color='forestgreen', linewidth=2, markersize=6, label='GPS точки')
plt.xlabel("Кумулятивна відстань (м)", fontsize=11)
plt.ylabel("Висота (м)", fontsize=11)
plt.title("Профіль висоти маршруту: Заросляк — Говерла", fontsize=13, fontweight='bold')

arrow_style = dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8)

info_text = (f"Дистанція: {distances[-1] / 1000:.1f} км\n"
             f"Набір висоти: {total_ascent:.0f} м\n"
             f"Макс. ухил: {np.max(grad_full):.1f}%")
plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, fontsize=11,
         verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='honeydew', edgecolor='forestgreen', alpha=0.9))

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(11, 6))
plt.plot(xx, yy_full, label="21 вузол (Еталон)", color='black', linewidth=2.5, alpha=0.8)
plt.plot(xx, yy_10, label="10 вузлів", color='red', linestyle='--', linewidth=2)
plt.plot(xx, yy_15, label="15 вузлів", color='blue', linestyle='-.', linewidth=2)
plt.plot(xx, yy_20, label="20 вузлів", color='green', linestyle=':', linewidth=2)
plt.title("Вплив кількості вузлів на точність сплайна", fontsize=13, fontweight='bold')
plt.xlabel("Відстань (м)", fontsize=11)
plt.ylabel("Висота (м)", fontsize=11)
plt.text(0.02, 0.98, "Висновок: більше вузлів =\nточніше відтворення рельєфу",
         transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='aliceblue', alpha=0.8))
plt.legend(loc='lower right', fontsize=10, frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(11, 5))
plt.plot(xx, err_10, label=f"10 вузлів (Макс: {np.max(err_10):.1f} м)", color='red')
plt.plot(xx, err_15, label=f"15 вузлів (Макс: {np.max(err_15):.1f} м)", color='blue')
plt.plot(xx, err_20, label=f"20 вузлів (Макс: {np.max(err_20):.1f} м)", color='green')
plt.title("Абсолютна похибка інтерполяції", fontsize=13, fontweight='bold')
plt.xlabel("Відстань (м)", fontsize=11)
plt.ylabel("Похибка (м)", fontsize=11)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()