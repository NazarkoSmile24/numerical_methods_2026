import math
import cmath



def F(x):
    return 3 * math.sin(x) - x

def dF(x):
    return 3 * math.cos(x) - 1

def d2F(x):
    return -3 * math.sin(x)

def tabulate_function(a, b, h, filename="lab8_tabulation.txt"):
    x = a
    results = []
    with open(filename, 'w') as f:
        f.write("x\t\tF(x)\n")
        f.write("-" * 30 + "\n")
        while x <= b + 1e-9:
            y = F(x)
            f.write(f"{x:.4f}\t\t{y:.4f}\n")
            results.append((x, y))
            x += h
    return results



def simple_iteration(x0, tau, eps):
    """Метод простої ітерації  """
    x_n = x0
    iterations = 0
    while True:
        iterations += 1
        x_next = x_n + tau * F(x_n)
        if abs(F(x_next)) < eps and abs(x_next - x_n) < eps: 
            break
        x_n = x_next
        if iterations > 10000: break
    return x_next, iterations

def newton_method(x0, eps):
    """Метод Ньютона"""
    x_n = x0
    iterations = 0
    while True:
        iterations += 1
        x_next = x_n - F(x_n) / dF(x_n) 
        if abs(F(x_next)) < eps and abs(x_next - x_n) < eps: 
            break
        x_n = x_next
        if iterations > 10000: break
    return x_next, iterations

def chebyshev_method(x0, eps):
    """Метод Чебишева"""
    x_n = x0
    iterations = 0
    while True:
        iterations += 1
        f_val, df_val, d2f_val = F(x_n), dF(x_n), d2F(x_n)
        term1 = f_val / df_val
        term2 = (f_val**2 * d2f_val) / (2 * df_val**3)
        x_next = x_n - term1 - term2 
        
        if abs(F(x_next)) < eps and abs(x_next - x_n) < eps:
            break
        x_n = x_next
        if iterations > 10000: break
    return x_next, iterations

def secant_method(x0, x1, eps):
    """Метод хорд"""
    iterations = 0
    while True:
        iterations += 1
        f_x1, f_x0 = F(x1), F(x0)
        x_next = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0) 
        if abs(F(x_next)) < eps and abs(x_next - x1) < eps: 
            break
        x0, x1 = x1, x_next
        if iterations > 10000: break
    return x_next, iterations

def parabola_method(x0, x1, x2, eps):
    """Метод парабол"""
    iterations = 0
    while True:
        iterations += 1
        f_x1_x0 = (F(x1) - F(x0)) / (x1 - x0)
        f_x2_x1 = (F(x2) - F(x1)) / (x2 - x1)
        A = (f_x2_x1 - f_x1_x0) / (x2 - x0)
        B = (x2 - x1) * A + f_x2_x1
        C = F(x2)
        

        disc = cmath.sqrt(B**2 - 4*A*C)
        denom1 = B + disc
        denom2 = B - disc

        if abs(denom1) > abs(denom2):
            delta = -2*C / denom1
        else:
            delta = -2*C / denom2
            
        x_next = x2 + delta.real 
        if abs(F(x_next)) < eps and abs(x_next - x2) < eps: 
            break
        x0, x1, x2 = x1, x2, x_next
        if iterations > 10000: break
    return x_next, iterations

def inverse_interpolation(x0, x1, x2, eps):
    """Метод зворотної інтерполяції"""
    iterations = 0
    while True:
        iterations += 1
        y0, y1, y2 = F(x0), F(x1), F(x2)
        term1 = (y1*y2) / ((y0-y1)*(y0-y2)) * x0
        term2 = (y0*y2) / ((y1-y0)*(y1-y2)) * x1
        term3 = (y0*y1) / ((y2-y0)*(y2-y1)) * x2
        x_next = term1 + term2 + term3 
        
        if abs(F(x_next)) < eps and abs(x_next - x2) < eps: 
            break
        x0, x1, x2 = x1, x2, x_next
        if iterations > 10000: break
    return x_next, iterations


def save_polynomial(coeffs, filename="lab8_poly_coeffs.txt"):
    with open(filename, 'w') as f:
        f.write(" ".join(str(c) for c in coeffs))

def read_polynomial(filename="lab8_poly_coeffs.txt"):
    with open(filename, 'r') as f:
        return [float(x) for x in f.read().split()]

def evaluate_poly(coeffs, x):
    res = 0
    m = len(coeffs) - 1
    for i, a in enumerate(coeffs):
        res += a * (x ** (m - i))
    return res

def newton_horner(coeffs, x0, eps):
    x_n = x0
    iterations = 0
    m = len(coeffs) - 1
    while True:
        iterations += 1
        b = [0] * (m + 1)
        b[0] = coeffs[0]
        for i in range(1, m + 1):
            b[i] = coeffs[i] + x_n * b[i-1]
        
        b_0 = b[-1]

        c = [0] * m
        c[0] = b[0]
        for i in range(1, m):
            c[i] = b[i] + x_n * c[i-1]
            
        c_1 = c[-1]
        
        x_next = x_n - b_0 / c_1 
        
        if abs(evaluate_poly(coeffs, x_next)) < eps and abs(x_next - x_n) < eps:
            break
        x_n = x_next
        if iterations > 1000: break
    return x_next, iterations

def lin_method(coeffs, alpha0, beta0, eps):

    a3, a2, a1, a0 = coeffs
    alpha_n, beta_n = alpha0, beta0
    iterations = 0
    
    while True:
        iterations += 1
        p0 = -2 * alpha_n 
        q0 = alpha_n**2 + beta_n**2
        
        b3 = a3
        b2 = a2 - p0 * b3 
        
        q1 = a0 / b2 
        p1 = (a1 * b2 - a0 * b3) / (b2**2)
        
        alpha_next = -p1 / 2 
        
   
        val = q1 - alpha_next**2
        if val < 0:
            val = abs(val) 
            
        beta_next = math.sqrt(val) 
        
        if abs(alpha_next - alpha_n) < eps and abs(beta_next - beta_n) < eps: 
            break
            
        alpha_n, beta_n = alpha_next, beta_next
        if iterations > 10000: break
        
    return complex(alpha_next, beta_next), iterations


if __name__ == "__main__":
    eps_target = 1e-10 
    
    print("--- 1. Трансцендентне рівняння F(x) = 3*sin(x) - x ---")
    tabulate_function(-1, 3, 0.1)
    print("Табуляцію збережено у 'lab8_tabulation.txt'.")
    
   
    roots_info = [
        {"name": "Корінь 1 (зростання)", "x0": 0.5, "tau": 0.4, "x1": 0.6, "x2": 0.7},
        {"name": "Корінь 2 (спадання)", "x0": 2.0, "tau": -0.2, "x1": 2.1, "x2": 2.2}
    ]
    
    for r in roots_info:
        print(f"\nДослідження: {r['name']}")
        x_simp, it_simp = simple_iteration(r['x0'], r['tau'], eps_target)
        print(f"  Проста ітерація: x = {x_simp:.10f}, ітерацій = {it_simp}")
        
        x_newt, it_newt = newton_method(r['x0'], eps_target)
        print(f"  Метод Ньютона:   x = {x_newt:.10f}, ітерацій = {it_newt}")
        
        x_cheb, it_cheb = chebyshev_method(r['x0'], eps_target)
        print(f"  Метод Чебишева:  x = {x_cheb:.10f}, ітерацій = {it_cheb}")
        
        x_sec, it_sec = secant_method(r['x0'], r['x1'], eps_target)
        print(f"  Метод хорд:      x = {x_sec:.10f}, ітерацій = {it_sec}")
        
        x_par, it_par = parabola_method(r['x0'], r['x1'], r['x2'], eps_target)
        print(f"  Метод парабол:   x = {x_par:.10f}, ітерацій = {it_par}")
        
        x_inv, it_inv = inverse_interpolation(r['x0'], r['x1'], r['x2'], eps_target)
        print(f"  Зворотна інтерп: x = {x_inv:.10f}, ітерацій = {it_inv}")

    print("\n--- 2. Алгебраїчне рівняння x^3 - 2x^2 + x - 2 = 0 ---")
    coeffs_orig = [1.0, -2.0, 1.0, -2.0]
    save_polynomial(coeffs_orig)
    print("Коефіцієнти збережено у 'lab8_poly_coeffs.txt'.\n")
    
    coeffs = read_polynomial()
    
    x_real, it_real = newton_horner(coeffs, x0=2.5, eps=eps_target)
    print(f"Дійсний корінь (Ньютон + Горнер): x = {x_real:.10f}, ітерацій = {it_real}")
    
    comp_root, it_comp = lin_method(coeffs, alpha0=0.1, beta0=0.9, eps=eps_target)
    print(f"Комплексний корінь (Ліна):      x = {comp_root.real:.10f} ± {comp_root.imag:.10f}i, ітерацій = {it_comp}\n")