import numpy as np
import matplotlib.pyplot as plt
import csv
import math

# ==========================================
# ЗАГАЛЬНІ МАТЕМАТИЧНІ ФУНКЦІЇ
# ==========================================

# 1. Розділені різниці та Метод Ньютона
def divided_differences(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:,0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
    return coef[0, :]

def newton_polynomial(coef, x_data, x_val):
    n = len(x_data) - 1 
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (x_val - x_data[n - k]) * p
    return p

# 2. Метод Лагранжа
def lagrange_polynomial(x_data, y_data, x_val):
    result = 0.0
    n = len(x_data)
    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if i != j:
                term = term * (x_val - x_data[j]) / (x_data[i] - x_data[j])
        result += term
    return result

# ==========================================
# ЧАСТИНА 1: ВАРІАНТ 3 (Модель машинного навчання)
# ==========================================
print("=== ЧАСТИНА 1: ВАРІАНТ 3 ===")

def read_data(filename):
    x, y = [], []
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['Dataset size']))
            y.append(float(row['Train time (sec)']))
    return np.array(x), np.array(y)

try:
    x_data, y_data = read_data("data.csv")
    coef_newton = divided_differences(x_data, y_data)
    x_target = 120000
    y_pred_newton = newton_polynomial(coef_newton, x_data, x_target)

    x_plot = np.linspace(min(x_data), max(x_data), 500)
    y_plot_newton = [newton_polynomial(coef_newton, x_data, xi) for xi in x_plot]

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot_newton, label='Метод Ньютона', color='blue', linewidth=2)
    plt.scatter(x_data, y_data, color='red', s=60, label='Експериментальні дані')
    plt.scatter([x_target], [y_pred_newton], color='cyan', s=120, marker='*', edgecolor='black', label=f'Прогноз ({y_pred_newton:.1f})')
    plt.title('Варіант 3: Прогноз часу тренування')
    plt.xlabel('Розмір датасету')
    plt.ylabel('Час тренування (сек)')
    plt.legend()
    plt.grid(True)
    # Звідси прибрано plt.show()
except FileNotFoundError:
    print("Файл data.csv не знайдено. Пропускаю ЧАСТИНУ 1.")

# ==========================================
# ЧАСТИНА 2: ДОСЛІДНИЦЬКА ЧАСТИНА
# ==========================================
print("=== ЧАСТИНА 2: ДОСЛІДНИЦЬКА ЧАСТИНА ===")
nodes_list = [5, 10, 20]
colors = ['blue', 'green', 'red']

# --- Пункт 1: Вплив кроку (Фіксований інтервал [0, 2pi], різна кількість вузлів) ---
def test_func_1(x): return np.sin(x)
a1, b1 = 0, 2 * np.pi
x_dense_1 = np.linspace(a1, b1, 1000)
y_true_1 = test_func_1(x_dense_1)

plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
plt.plot(x_dense_1, y_true_1, 'k-', linewidth=2, label='Справжня функція sin(x)')
for n, c in zip(nodes_list, colors):
    x_nodes = np.linspace(a1, b1, n)
    y_nodes = test_func_1(x_nodes)
    coef = divided_differences(x_nodes, y_nodes)
    y_pred = [newton_polynomial(coef, x_nodes, xi) for xi in x_dense_1]
    plt.plot(x_dense_1, y_pred, color=c, label=f'Ньютон (n={n}, крок={(b1-a1)/(n-1):.2f})')
plt.title('1. Вплив кроку (Фіксований інтервал [0, 2π])')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
for n, c in zip(nodes_list, colors):
    x_nodes = np.linspace(a1, b1, n)
    y_nodes = test_func_1(x_nodes)
    coef = divided_differences(x_nodes, y_nodes)
    y_pred = [newton_polynomial(coef, x_nodes, xi) for xi in x_dense_1]
    error = np.abs(y_true_1 - np.array(y_pred))
    plt.plot(x_dense_1, error, color=c, label=f'Похибка (n={n})')
plt.title('Абсолютні похибки (Логарифмічна шкала)')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Звідси прибрано plt.show()

# --- Пункт 2: Вплив кількості вузлів (Фіксований крок h=0.5, змінний інтервал) ---
h = 0.5
plt.figure(figsize=(12, 6))
for n, c in zip(nodes_list, colors):
    x_nodes = np.array([i * h for i in range(n)]) # x_0 = 0
    y_nodes = test_func_1(x_nodes)
    coef = divided_differences(x_nodes, y_nodes)
    
    # Інтервал для оцінки [0, (n-1)*h]
    x_dense_2 = np.linspace(0, (n-1)*h, 500)
    y_true_2 = test_func_1(x_dense_2)
    y_pred = [newton_polynomial(coef, x_nodes, xi) for xi in x_dense_2]
    
    error = np.abs(y_true_2 - np.array(y_pred))
    plt.plot(x_dense_2, error, color=c, label=f'Похибка n={n} (Інтервал [0, {(n-1)*h}])')

plt.title('2. Вплив кількості вузлів (Фіксований крок h=0.5, змінний інтервал)')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('Абсолютна похибка')
plt.legend()
plt.grid(True)
# Звідси прибрано plt.show()

# --- Пункт 3: Аналіз ефекту Рунге ---
def runge_function(x): return 1 / (1 + 25 * x**2)
x_dense_runge = np.linspace(-1, 1, 1000)
y_true_runge = runge_function(x_dense_runge)

plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
plt.plot(x_dense_runge, y_true_runge, 'k-', linewidth=2, label='Справжня функція f(x)')
for n, c in zip(nodes_list, colors):
    x_nodes = np.linspace(-1, 1, n)
    y_nodes = runge_function(x_nodes)
    coef = divided_differences(x_nodes, y_nodes)
    y_pred = [newton_polynomial(coef, x_nodes, xi) for xi in x_dense_runge]
    plt.plot(x_dense_runge, y_pred, color=c, label=f'Ньютон (n={n})')
plt.title('3. Ефект Рунге: Інтерполяція многочленами Ньютона')
plt.ylim(-1, 2)
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
for n, c in zip(nodes_list, colors):
    x_nodes = np.linspace(-1, 1, n)
    y_nodes = runge_function(x_nodes)
    coef = divided_differences(x_nodes, y_nodes)
    y_pred = [newton_polynomial(coef, x_nodes, xi) for xi in x_dense_runge]
    error = np.abs(y_true_runge - np.array(y_pred))
    plt.plot(x_dense_runge, error, color=c, label=f'Похибка (n={n})')
plt.title('Абсолютні похибки (Ефект Рунге)')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Звідси прибрано plt.show()

# --- Пункт 4: Порівняння з методом Лагранжа ---
# Використаємо n=10 для функції Рунге
n_lagrange = 10
x_nodes_l = np.linspace(-1, 1, n_lagrange)
y_nodes_l = runge_function(x_nodes_l)

coef_n = divided_differences(x_nodes_l, y_nodes_l)
y_pred_newton_l = [newton_polynomial(coef_n, x_nodes_l, xi) for xi in x_dense_runge]
y_pred_lagrange_l = [lagrange_polynomial(x_nodes_l, y_nodes_l, xi) for xi in x_dense_runge]

diff_nl = np.abs(np.array(y_pred_newton_l) - np.array(y_pred_lagrange_l))

plt.figure(figsize=(10, 5))
plt.plot(x_dense_runge, diff_nl, 'r-', label='|Ньютон - Лагранж|')
plt.title('4. Порівняння Ньютона та Лагранжа (n=10)')
plt.xlabel('x')
plt.ylabel('Абсолютна різниця')
plt.yscale('log')
plt.legend()
plt.grid(True)

print("Усі розрахунки виконано! Відкриваю 5 вікон із графіками одночасно...")

# ЄДИНИЙ виклик plt.show() наприкінці коду, що відображає усі згенеровані фігури
plt.show()