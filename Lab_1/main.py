import requests
import numpy as np
import matplotlib.pyplot as plt

url = ("https://api.open-elevation.com/api/v1/lookup?locations="
       "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
       "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
       "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|"
       "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
       "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
       "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
       "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106")

try:
    response = requests.get(url)
    data = response.json()
    results = data["results"]
except Exception as e:
    exit()

n_total = len(results)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

coords_all = [(p["latitude"], p["longitude"]) for p in results]
elevations_all = np.array([p["elevation"] for p in results])
distances_all = [0.0]

for i in range(1, n_total):
    d = haversine(*coords_all[i-1], *coords_all[i])
    distances_all.append(distances_all[-1] + d)
distances_all = np.array(distances_all)

def solve_cubic_spline(x, y):
    n = len(x)
    h = []
    for i in range(1, n):
        h.append(x[i] - x[i-1])
    alpha = np.zeros(n)
    beta = np.zeros(n)
    gamma = np.zeros(n)
    delta = np.zeros(n)
    
    beta[0] = 1.0
    for i in range(1, n-1):
        alpha[i] = h[i-1]
        beta[i] = 2 * (h[i-1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3 * ((y[i+1] - y[i])/h[i] - (y[i] - y[i-1])/h[i-1])
    beta[n-1] = 1.0 

    A = np.zeros(n)
    B = np.zeros(n)
    A[0] = -gamma[0] / beta[0]
    B[0] = delta[0] / beta[0]
    
    for i in range(1, n):
        denom = alpha[i] * A[i-1] + beta[i]
        if i < n-1:
            A[i] = -gamma[i] / denom
        B[i] = (delta[i] - alpha[i] * B[i-1]) / denom
        
    c = np.zeros(n)
    c[n-1] = B[n-1]
    for i in range(n-2, -1, -1):
        c[i] = A[i] * c[i+1] + B[i]
        
    a = y[:-1]
    b = []
    d = []
    for i in range(n-1):
        hi = h[i]
        b.append((y[i+1] - y[i])/hi - hi*(c[i+1] + 2*c[i])/3)
        d.append((c[i+1] - c[i])/(3 * hi))
        
    return a, b, c, d

def get_spline_val(val, x_pts, a, b, c, d):
    if val <= x_pts[0]: return a[0]
    if val >= x_pts[-1]: return elevations_all[-1]
    for i in range(len(x_pts) - 1):
        if x_pts[i] <= val <= x_pts[i+1]:
            dx = val - x_pts[i]
            return a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
    return 0

node_counts = [10, 15, 21]
plt.figure(figsize=(12, 6))

for count in node_counts:
    indices = np.linspace(0, n_total - 1, count, dtype=int)
    x_nodes = distances_all[indices]
    y_nodes = elevations_all[indices]
    a_c, b_c, c_c, d_c = solve_cubic_spline(x_nodes, y_nodes)
    x_fine = np.linspace(x_nodes[0], x_nodes[-1], 500)
    y_spline = [get_spline_val(v, x_nodes, a_c, b_c, c_c, d_c) for v in x_fine]
    plt.plot(x_fine, y_spline, label=f'n={count}')

plt.scatter(distances_all, elevations_all, color='black', s=10)
plt.title('Cubic Spline Interpolation')
plt.xlabel('Distance (m)')
plt.ylabel('Elevation (m)')
plt.legend()
plt.grid(True)
plt.show()

idx_err = np.linspace(0, n_total - 1, 10, dtype=int)
xn_err, yn_err = distances_all[idx_err], elevations_all[idx_err]
ae, be, ce, de = solve_cubic_spline(xn_err, yn_err)

y_approx = np.array([get_spline_val(d, xn_err, ae, be, ce, de) for d in distances_all])
error = np.abs(elevations_all - y_approx)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
ax1.plot(distances_all, elevations_all, 'k--', label='f(x)')
ax1.plot(distances_all, y_approx, 'b-', label='S(x)')
ax1.set_ylabel('Elevation (m)')
ax1.legend()
ax1.grid(True)

ax2.plot(distances_all, error, 'r-o', label='Error')
ax2.set_xlabel('Distance (m)')
ax2.set_ylabel('Absolute Error (m)')
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.show()

total_ascent = sum(max(elevations_all[i] - elevations_all[i-1], 0) for i in range(1, n_total))
total_descent = sum(max(elevations_all[i-1] - elevations_all[i], 0) for i in range(1, n_total))
energy = 80 * 9.81 * total_ascent

print(f"Distance: {distances_all[-1]:.2f} m")
print(f"Ascent: {total_ascent:.2f} m")
print(f"Descent: {total_descent:.2f} m")
print(f"Work: {energy/1000:.2f} kJ")