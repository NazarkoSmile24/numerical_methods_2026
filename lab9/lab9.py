import math
import numpy as np
import matplotlib.pyplot as plt

# 1. Тестові та Цільові функції

def rosenbrock(X):
    x1, x2 = X[0], X[1]
    return 100 * (x1**2 - x2)**2 + (x1 - 1)**2

def non_linear_system(X):
    x1, x2 = X[0], X[1]
    f1 = x2 - x1**2
    f2 = x1 - x2
    return f1**2 + f2**2

# 2. Алгоритм Хука-Дживса

def exploratory_search(X_start, delta_X, reduce_step, q, eps1, func):
    X_curr = list(X_start)
    n = len(X_curr)
    
    for i in range(n):
        while True:
            X_forward = list(X_curr)
            X_forward[i] += delta_X[i]  # рух вперед
            if func(X_forward) < func(X_curr):
                X_curr = X_forward
                break
                
            X_backward = list(X_curr)
            X_backward[i] -= delta_X[i]  # рух назад

            # перевірка, якщо функція стала меншою то переходимо у нову точку

            if func(X_backward) < func(X_curr):
                X_curr = X_backward
                break
                
            # якщо покращення немає, то зменшуємо крок
            
            if reduce_step:
                delta_X[i] /= q
                if delta_X[i] < eps1:
                    break
            else:
                break
                
    return X_curr, delta_X

def hooke_jeeves(func, X0, initial_delta, q=2.0, p=2.0, eps1=1e-6, eps2=1e-6):
    X_0 = list(X0)
    delta_X = list(initial_delta)
    trajectory = [list(X_0)]
    
    while True:
        X_1, delta_X = exploratory_search(X_0, delta_X, reduce_step=True, q=q, eps1=eps1, func=func)
        
        if X_1 != X_0:
            trajectory.append(list(X_1))
            max_delta = max(delta_X)
            diff_f = abs(func(X_1) - func(X_0))
            
            # завершення алгоритму

            if max_delta < eps1 and diff_f < eps2:
                X_0 = X_1
                break
                
            # якщо знайдено хороший напрям, ьто метод робить більший крок у цьому напрямку

            while True:
                X_p = [X_1[i] + p * (X_1[i] - X_0[i]) for i in range(len(X_0))]
                X_2, _ = exploratory_search(X_p, delta_X, reduce_step=False, q=q, eps1=eps1, func=func)
                
                if func(X_2) < func(X_1):
                    X_0 = list(X_1)
                    X_1 = list(X_2)
                    trajectory.append(list(X_1))
                else:
                    X_0 = list(X_1)
                    break
        else:
            break
            
    return X_0, trajectory

# Допоміжні функції 

def save_trajectory(filename, trajectory, func_name):
    with open(filename, 'w') as f:
        f.write("step\tX1\t\tX2\n")
        f.write("-" * 40 + "\n")
        for idx, point in enumerate(trajectory):
            f.write(f"{idx}\t{point[0]:.6f}\t{point[1]:.6f}\n")

def plot_system_with_trajectory(trajectory, root_X):

    x1_parabola = np.linspace(-3, 3, 400)
    x2_parabola = x1_parabola**2

    x1_line = np.linspace(-3, 3, 100)
    x2_line = x1_line

    plt.figure(figsize=(9, 9))

    plt.plot(
        x1_parabola,
        x2_parabola,
        label='$x_2 = x_1^2$ (Парабола)',
        color='blue',
        markersize=7,
        linewidth=2,
        alpha=0.6
    )

    plt.plot(
        x1_line,
        x2_line,
        label='$x_1 - x_2 = 0$ (Пряма)',
        color='red',
        markersize=7,
        linewidth=2,
        alpha=0.6
    )

    if trajectory:
        t_x1 = [p[0] for p in trajectory]
        t_x2 = [p[1] for p in trajectory]
        plt.plot(t_x1, t_x2, label='Траєкторія Хука-Дживса', color='green', marker='o', linestyle='--', markersize=5)
        
        plt.scatter(t_x1[0], t_x2[0], color='orange', s=120, label='Початкова точка', zorder=6, edgecolors='black')
    plt.scatter(root_X[0], root_X[1], color='black', s=120, label='Знайдений розв\'язок', zorder=7)
    plt.text(root_X[0] + 0.1, root_X[1] - 0.2, f'({root_X[0]:.3f}, {root_X[1]:.3f})', fontsize=11, fontweight='bold')

    plt.axhline(0, color='black', linewidth=1.0)
    plt.axvline(0, color='black', linewidth=1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.legend(loc='upper left', fontsize=11)
    plt.title('Метод Хука-Дживса: Розв\'язок системи нелінійних рівнянь', fontsize=14, pad=15)
    plt.xlabel('$x_1$', fontsize=12)
    plt.ylabel('$x_2$', fontsize=12)
    
    plt.axis('equal') 
    plt.xlim(-3.5, 3.5)
    plt.ylim(-3.5, 3.5)
    plt.show()

# 4. Головний блок програми

if __name__ == "__main__":
    q, p = 2.0, 2.0
    eps1, eps2 = 1e-6, 1e-6

    print("--- 1. Тестування на функції Розенброка ---")
    X0_rosen = [-1.2, 0.0]
    delta_rosen = [0.5, 0.5]
    res_rosen, traj_rosen = hooke_jeeves(rosenbrock, X0_rosen, delta_rosen, q, p, eps1, eps2)
    save_trajectory("trajectory_rosenbrock.txt", traj_rosen, "Функції Розенброка")
    
    print(f"Початкова точка: {X0_rosen}")
    print(f"Знайдений мінімум: X* = [{res_rosen[0]:.6f}, {res_rosen[1]:.6f}]")
    print(f"Значення функції: Ф(X*) = {rosenbrock(res_rosen):.10e}")
    print(f"Кількість кроків траєкторії: {len(traj_rosen)}")
    

    print("\n--- 2. Розв'язок системи нелінійних рівнянь ---")
    X0_sys = [1.0, 0.0] 
    delta_sys = [0.5, 0.5]
    
    res_sys, traj_sys = hooke_jeeves(non_linear_system, X0_sys, delta_sys, q, p, eps1, eps2)
    save_trajectory("trajectory_system.txt", traj_sys, "Системи нелінійних рівнянь")
    
    print(f"Початкова точка: {X0_sys}")
    print(f"Знайдений розв'язок: X* = [{res_sys[0]:.6f}, {res_sys[1]:.6f}]")
    print(f"Значення цільової функції Ф(X*): {non_linear_system(res_sys):.10e}")
    print(f"Кількість кроків траєкторії: {len(traj_sys)}")

    plot_system_with_trajectory(traj_sys, res_sys)