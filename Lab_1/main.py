import requests
import numpy as np
import matplotlib.pyplot as plt
import json

url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
response = requests.get(url)
data = response.json()

results = data["results"]
n = len(results)

print("Кількість вузлів", n)    

print("Табуляція вузлів:")
print("№ | Latitude | Longitude | Elevation (m)")
for i, point in enumerate(results):
    print(f"{i:2d} | {point['latitude']:.6f} | "
    f"{point['longitude']:.6f} | "
    f"{point['elevation']:.2f}")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]

distances = [0]
for i in range(1, n):
    d = haversine(*coords[i-1], *coords[i])
    distances.append(distances[-1] + d)

print("\nТабуляція (відстань, висота):")
print("№ | Distance (m) | Elevation (m)")
for i in range(n):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")

def solve_spline(x, y):
    n = len(x)
    h = [x[i] - x[i-1] for i in range(1, n)]
    
    alpha = [0] * n
    beta = [0] * n
    gamma = [0] * n
    delta = [0] * n
    
    beta[0] = 1
    for i in range(1, n-1):
        alpha[i] = h[i-1]
        beta[i] = 2 * (h[i-1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3 * ((y[i+1] - y[i])/h[i] - (y[i] - y[i-1])/h[i-1])
    beta[n-1] = 1

    A = [0] * n
    B = [0] * n
    A[0] = -gamma[0] / beta[0]
    B[0] = delta[0] / beta[0]
    
    for i in range(1, n-1):
        A[i] = -gamma[i] / (alpha[i] * A[i-1] + beta[i])
        B[i] = (delta[i] - alpha[i] * B[i-1]) / (alpha[i] * A[i-1] + beta[i])
        
    c = [0] * n
    c[n-1] = (delta[n-1] - alpha[n-1] * B[n-2]) / (alpha[n-1] * A[n-2] + beta[n-1])
    for i in range(n-2, -1, -1):
        c[i] = A[i] * c[i+1] + B[i]
        
    a = y[:-1]
    b = []
    d = []
    for i in range(n-1):
        b.append((y[i+1] - y[i])/h[i] - h[i]*(c[i+1] + 2*c[i])/3)
        d.append((c[i+1] - c[i])/(3 * h[i]))
        
    return a, b, c[:-1], d

a_coeffs, b_coeffs, c_coeffs, d_coeffs = solve_spline(distances, elevations)

def interpolate(val, x_points, a, b, c, d):
    for i in range(len(x_points) - 1):
        if x_points[i] <= val <= x_points[i+1]:
            dx = val - x_points[i]
            return a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
    return 0

x_fine = np.linspace(distances[0], distances[-1], 200)
y_fine = [interpolate(val, distances, a_coeffs, b_coeffs, c_coeffs, d_coeffs) for val in x_fine]

total_ascent = sum(max(elevations[i] - elevations[i-1], 0) for i in range(1, n))
total_descent = sum(max(elevations[i-1] - elevations[i], 0) for i in range(1, n))
energy = 80 * 9.81 * total_ascent

print(f"Загальна довжина: {distances[-1]:.2f} м")
print(f"Сумарний набір: {total_ascent:.2f} м")
print(f"Механічна робота: {energy/1000:.2f} кДж")

plt.figure(figsize=(10, 6))
plt.plot(distances, elevations, 'ro', label='Вузли')
plt.plot(x_fine, y_fine, 'b-', label='Кубічний сплайн')
plt.title('Профіль висоти: Заросляк - Говерла')
plt.xlabel('Відстань (м)')
plt.ylabel('Висота (м)')
plt.legend()
plt.grid(True)
plt.show()