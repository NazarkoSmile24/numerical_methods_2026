import csv
import importlib.util
import os
from typing import Dict, List, Tuple

if importlib.util.find_spec("matplotlib") is not None:
    import matplotlib.pyplot as plt
else:
    plt = None


EPS = 1e-12


def read_csv(filename: str) -> Tuple[List[float], List[float]]:
    """Read Month/Temp columns from CSV and return x, y lists."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл '{filename}' не знайдено.")

    with open(filename, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV-файл порожній або не містить заголовка.")

        fields = [name.strip() for name in reader.fieldnames]
        if "Month" not in fields or "Temp" not in fields:
            raise ValueError("CSV повинен містити стовпці 'Month' і 'Temp'.")

        x: List[float] = []
        y: List[float] = []

        for idx, row in enumerate(reader, start=2):
            try:
                month_raw = row.get("Month")
                temp_raw = row.get("Temp")
                if month_raw is None or temp_raw is None:
                    raise ValueError

                month = float(month_raw)
                temp = float(temp_raw)
            except (TypeError, ValueError):
                raise ValueError(f"Неправильний формат даних у рядку {idx}: {row}") from None

            x.append(month)
            y.append(temp)

    if not x:
        raise ValueError("CSV-файл не містить жодного рядка даних.")

    if len(x) < 2:
        raise ValueError("Недостатньо даних для апроксимації (потрібно мінімум 2 точки).")

    return x, y


def form_matrix(x: List[float], m: int) -> List[List[float]]:
    """Create normal-equation matrix A for polynomial degree m."""
    return [
        [sum((xk ** (i + j)) for xk in x) for j in range(m + 1)]
        for i in range(m + 1)
    ]


def form_vector(x: List[float], y: List[float], m: int) -> List[float]:
    """Create right-hand side vector b for polynomial degree m."""
    return [sum(yk * (xk ** i) for xk, yk in zip(x, y)) for i in range(m + 1)]


def gauss_solve(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve linear system Ax=b using Gaussian elimination with partial pivoting by column."""
    n = len(A)

    # Work on copies so the original data is preserved.
    a = [row[:] for row in A]
    rhs = b[:]

    # Forward elimination.
    for k in range(n):
        pivot_row = max(range(k, n), key=lambda i: abs(a[i][k]))
        pivot_value = a[pivot_row][k]

        if abs(pivot_value) < EPS:
            raise ValueError("Система вироджена або майже вироджена (нульовий pivot).")

        if pivot_row != k:
            a[k], a[pivot_row] = a[pivot_row], a[k]
            rhs[k], rhs[pivot_row] = rhs[pivot_row], rhs[k]

        for i in range(k + 1, n):
            factor = a[i][k] / a[k][k]
            a[i][k] = 0.0
            for j in range(k + 1, n):
                a[i][j] -= factor * a[k][j]
            rhs[i] -= factor * rhs[k]

    # Back substitution.
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(a[i][j] * x[j] for j in range(i + 1, n))
        denom = a[i][i]
        if abs(denom) < EPS:
            raise ValueError("Система не має єдиного розв'язку (нуль на діагоналі).")
        x[i] = (rhs[i] - s) / denom

    return x


def polynomial(x_value: float, coef: List[float]) -> float:
    """Evaluate polynomial at x_value."""
    return sum(coef[i] * (x_value ** i) for i in range(len(coef)))


def variance(y_true: List[float], y_approx: List[float]) -> float:
    """Mean squared error for two equally sized vectors."""
    if len(y_true) != len(y_approx):
        raise ValueError("Списки y_true та y_approx повинні бути однакової довжини.")
    n = len(y_true)
    return sum((y_true[i] - y_approx[i]) ** 2 for i in range(n)) / n


def find_best_degree(
    x: List[float], y: List[float], max_degree: int = 10
) -> Tuple[int, Dict[int, Dict[str, List[float] | float]]]:
    """Compute least-squares approximations for degrees 1..max_degree and choose best."""
    if max_degree < 1:
        raise ValueError("max_degree повинен бути >= 1")

    results: Dict[int, Dict[str, List[float] | float]] = {}

    for m in range(1, max_degree + 1):
        A = form_matrix(x, m)
        b = form_vector(x, y, m)
        coef = gauss_solve(A, b)

        y_approx = [polynomial(xk, coef) for xk in x]
        err = variance(y, y_approx)
        errors = [y[i] - y_approx[i] for i in range(len(y))]

        results[m] = {
            "coef": coef,
            "y_approx": y_approx,
            "variance": err,
            "errors": errors,
        }

    best_degree = min(results, key=lambda deg: results[deg]["variance"])
    return best_degree, results


def plot_results(
    x: List[float],
    y: List[float],
    results: Dict[int, Dict[str, List[float] | float]],
    best_degree: int,
    forecast_months: List[float],
    forecast_values: List[float],
) -> None:

    if plt is None:
        raise RuntimeError("matplotlib не встановлено. Побудова графіків недоступна.")

    degrees = sorted(results.keys())
    variances = [float(results[d]["variance"]) for d in degrees]

    plt.figure(figsize=(8, 5))
    plt.plot(degrees, variances, marker="o", label="Середньоквадратична похибка")
    plt.title("Залежність похибки від степеня полінома")
    plt.xlabel("Степінь полінома")
    plt.ylabel("Похибка (D)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    best_coef = results[best_degree]["coef"]
    best_errors = results[best_degree]["errors"]

    x_min = min(x)
    x_max = max(forecast_months)
    points_count = 300
    step = (x_max - x_min) / (points_count - 1) if points_count > 1 else 1.0
    x_curve = [x_min + i * step for i in range(points_count)]
    y_curve = [polynomial(x_value, best_coef) for x_value in x_curve]

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, "o", label="Табличні дані")
    plt.plot(x_curve, y_curve, "-", label=f"Апроксимація (m={best_degree})")
    plt.plot(forecast_months, forecast_values, "ro", label="Прогноз (наступні 3 місяці)")
    plt.title("Вихідні дані та поліном найкращого наближення")
    plt.xlabel("Місяць")
    plt.ylabel("Температура")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 5))
    plt.axhline(0.0, color="black", linewidth=1)
    plt.plot(x, best_errors, marker="o", label=f"Похибки (m={best_degree})")
    plt.title("Похибки апроксимації по точках")
    plt.xlabel("Місяць")
    plt.ylabel("y - y_approx")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


def print_error_table(x: List[float], y: List[float], y_approx: List[float]) -> None:
    """Print point-by-point approximation errors for the best polynomial."""
    print("\nТаблиця похибок апроксимації (найкращий поліном):")
    print(f"{'Month':>7} | {'Actual':>10} | {'Approx':>10} | {'Error':>10}")
    print("-" * 49)

    for month, actual, approx in zip(x, y, y_approx):
        error = actual - approx
        print(f"{month:7.0f} | {actual:10.3f} | {approx:10.3f} | {error:10.3f}")


def main() -> None:
    filename = "data.csv"

    try:
        x, y = read_csv(filename)

        best_degree, results = find_best_degree(x, y, max_degree = 10)

        for m in sorted(results):
            print(f"Степінь {m}: похибка = {results[m]['variance']:.6f}")

        best_coef = results[best_degree]["coef"]
        print(f"\nНайкращий степінь: {best_degree}")
        print("Коефіцієнти полінома:")
        for i, c in enumerate(best_coef):
            print(f"c{i} = {c:.6f}")

        forecast_months = [25, 26, 27]
        forecast_values = [polynomial(month, best_coef) for month in forecast_months]

        best_y_approx = results[best_degree]["y_approx"]
        print_error_table(x, y, best_y_approx)

        print("\nПрогноз:")
        for month, value in zip(forecast_months, forecast_values):
            print(f"{month} -> {value:.6f}")

        plot_results(x, y, results, best_degree, forecast_months, forecast_values)

    except FileNotFoundError as e:
        print(f"Помилка: {e}")
    except ValueError as e:
        print(f"Помилка даних: {e}")
    except RuntimeError as e:
        print(f"Попередження: {e}")
    except Exception as e:
        print(f"Непередбачена помилка: {e}")


if __name__ == "__main__":
    main()
