'''import matplotlib.pyplot as plt
import numpy as np

def fixed_point(g, x0, tol=1e-8, iterations=100):
    it = 0
    x_current = x0
    x_next = g(x0)
    while np.abs(x_next-x_current) > tol and it < iterations:
        x_current = x_next
        x_next = g(x_current)
        it += 1
    return x_next
    
def crank_nicolson(T, f, y0, N):
    h = T/(N+1)
    t = np.linspace(0, T, N+2)
    z = np.zeros(N+2)
    z[0] = y0
    for k in range(N+1):
        def g(x):
            return z[k] + (h/2)* (f(t[k], z[k]) + f(t[k+1], x))
        z[k+1] = fixed_point(g, z[k])
    return t, z

def f(t,y):
    return t - y

T = 1
y0 = 0
N = 100


t, z_crank_nicolson = crank_nicolson(T, f, y0, N)
z_truth = np.exp(-t) + t-1

plt.plot(t, z_crank_nicolson, color = "blue")
plt.plot(t, z_truth, color = "red", linestyle='--')
plt.show()

plt.plot(t, z_crank_nicolson, color="blue", label="Crank-Nicolson")
plt.plot(t, z_truth, 'r--', label="Exact Solution")  # Dashed red line
plt.xlabel("Time t")
plt.ylabel("Solution z")
plt.legend()
plt.grid()
plt.show()

error_crank_nicolson = np.sum((z_truth - z_crank_nicolson) ** 2)
print(error_crank_nicolson)

'''


import numpy as np
import matplotlib.pyplot as plt

def fixed_point_vector(g, x0, tol=1e-8, iterations=100):
    """Fixed-point iteration for vector-valued functions."""
    it = 0
    x_current = x0
    x_next = g(x0)
    while np.linalg.norm(x_next - x_current) > tol and it < iterations:
        x_current = x_next
        x_next = g(x_current)
        it += 1
    return x_next

def crank_nicolson(T, f, y0, N):
    """Crank-Nicolson method for vector-valued functions in R^d."""
    h = T / (N + 1)
    t = np.linspace(0, T, N+2)
    d = len(y0)  # Dimension of the system
    z = np.zeros((N+2, d))  # Store solutions as an array of vectors
    z[0] = y0

    for k in range(N+1):
        def g(x):
            return z[k] + (h/2) * (f(t[k], z[k]) + f(t[k+1], x))

        z[k+1] = fixed_point_vector(g, z[k])  # Solve for the next vector z[k+1]

    return t, z
