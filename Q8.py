import math
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
 
def f(x):
    return x**2 - 6*x + 13

# ------------------------------------------------------------
# Simulated Annealing main loop
# ------------------------------------------------------------
random_numbers = [0.30, 0.80, 0.20, 0.90]
r_index = 0                 # pointer, advances only when Delta f > 0
 
T = 10.0
alpha = 0.5
T_min = 1.0
 
x_current = 1
best_x = x_current
best_f = f(x_current)
 
table_rows = []
 
for iteration in range(1, 5):
    x_prime = x_current + 1              # neighbour generation
    fx  = f(x_current)
    fxp = f(x_prime)
    delta_f = fxp - fx
 
    if delta_f <= 0:
        P = 1.0
        r_used = "-"
        accept = True
    else:
        P = math.exp(-delta_f / T)
        r_used = random_numbers[r_index]
        r_index += 1
        accept = r_used < P
 
    new_x = x_prime if accept else x_current
 
    if f(new_x) < best_f:
        best_f = f(new_x)
        best_x = new_x
 
    table_rows.append({
        "Iteration": iteration, "T": T, "Current x": x_current,
        "Neighbour x'": x_prime, "f(x)": fx, "f(x')": fxp,
        "Delta f": delta_f, "P": round(P, 4), "r": r_used,
        "Accept/Reject": "Accept" if accept else "Reject",
        "Best solution (x, f(x))": f"({best_x}, {best_f})"
    })
 
    x_current = new_x
    T = T * alpha
 
sa_table = pd.DataFrame(table_rows)
print(sa_table)
print(f"\nBest x found by Simulated Annealing = {best_x}")
print(f"Best f(x) found by Simulated Annealing = {best_f}")
 
# ------------------------------------------------------------
# Analytical verification
# ------------------------------------------------------------
x = sp.symbols('x')
f_sym = x**2 - 6*x + 13
f_prime = sp.diff(f_sym, x)
critical_points = sp.solve(sp.Eq(f_prime, 0), x)
x_star = critical_points[0]
f_star = f_sym.subs(x, x_star)
print(f"\nAnalytical minimum: x* = {x_star}, f(x*) = {f_star}")
