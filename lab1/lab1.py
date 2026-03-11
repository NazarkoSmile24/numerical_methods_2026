import argparse
from pathlib import Path

import numpy as np
import requests


COORDS = [
    (48.164214, 24.536044),
    (48.164983, 24.534836),
    (48.165605, 24.534068),
    (48.166228, 24.532915),
    (48.166777, 24.531927),
    (48.167326, 24.530884),
    (48.167011, 24.530061),
    (48.166053, 24.528039),
    (48.166655, 24.526064),
    (48.166497, 24.523574),
    (48.166128, 24.520214),
    (48.165416, 24.517170),
    (48.164546, 24.514640),
    (48.163412, 24.512980),
    (48.162331, 24.511715),
    (48.162015, 24.509462),
    (48.162147, 24.506932),
    (48.161751, 24.504244),
    (48.161197, 24.501793),
    (48.160580, 24.500537),
    (48.160250, 24.500106),
]


def build_open_elevation_url(coords: list[tuple[float, float]]) -> str:
    joined = "|".join([f"{lat:.6f},{lon:.6f}" for lat, lon in coords])
    return f"https://api.open-elevation.com/api/v1/lookup?locations={joined}"


def haversine(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def cumulative_distances(coords: list[tuple[float, float]]) -> np.ndarray:
    d = [0.0]
    for i in range(1, len(coords)):
        d.append(d[-1] + haversine(*coords[i - 1], *coords[i]))
    return np.array(d, dtype=float)


def thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = len(b)
    if len(d) != n:
        raise ValueError("Розмір d має дорівнювати розміру b")
    if len(a) != n - 1 or len(c) != n - 1:
        raise ValueError("Розміри a,c мають бути n-1")

    ac = a.astype(float).copy()
    bc = b.astype(float).copy()
    cc = c.astype(float).copy()
    dc = d.astype(float).copy()

    for i in range(1, n):
        if abs(bc[i - 1]) < 1e-15:
            raise ZeroDivisionError("Ділення на нуль у прогонці (bc[i-1]≈0)")
        w = ac[i - 1] / bc[i - 1]
        bc[i] = bc[i] - w * cc[i - 1]
        dc[i] = dc[i] - w * dc[i - 1]

    x = np.zeros(n, dtype=float)
    if abs(bc[-1]) < 1e-15:
        raise ZeroDivisionError("Ділення на нуль у прогонці (bc[-1]≈0)")
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        if abs(bc[i]) < 1e-15:
            raise ZeroDivisionError("Ділення на нуль у прогонці (bc[i]≈0)")
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]
    return x


class NaturalCubicSpline:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("x та y мають бути 1D та однакової довжини")
        if len(x) < 2:
            raise ValueError("Потрібно мінімум 2 вузли")
        if not np.all(np.diff(x) > 0):
            raise ValueError("x має бути строго зростаючим")

        self.x = x
        self.y = y
        self.n = len(x) - 1

        self.a = y[:-1].copy()
        self.b = np.zeros(self.n)
        self.c = np.zeros(self.n + 1)
        self.d = np.zeros(self.n)

        self._build()

    def _build(self):
        x, y, n = self.x, self.y, self.n
        h = np.diff(x)

        if n == 1:
            self.b[0] = (y[1] - y[0]) / h[0]
            self.c[:] = 0.0
            self.d[0] = 0.0
            return

        # m = n - 1 внутрішніх вузлів для c_1..c_{n-1}
        a_sub = h[1:-1].copy()
        b_diag = 2 * (h[:-1] + h[1:])
        c_sup = h[1:-1].copy()
        rhs = 3 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])

        c_inner = thomas(a_sub, b_diag, c_sup, rhs)
        self.c[0] = 0.0
        self.c[n] = 0.0
        self.c[1:n] = c_inner

        for i in range(n):
            self.d[i] = (self.c[i + 1] - self.c[i]) / (3 * h[i])
            self.b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * self.c[i] + self.c[i + 1]) / 3

    def eval(self, xq: np.ndarray) -> np.ndarray:
        xq = np.asarray(xq, dtype=float)
        x, n = self.x, self.n

        idx = np.searchsorted(x, xq, side="right") - 1
        idx = np.clip(idx, 0, n - 1)

        dx = xq - x[idx]
        return self.a[idx] + self.b[idx] * dx + self.c[idx] * dx**2 + self.d[idx] * dx**3


def pick_evenly(x: np.ndarray, y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    if k < 2 or k > n:
        raise ValueError("k має бути в межах [2, len(x)]")
    idx = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))
    while len(idx) < k:
        missing = k - len(idx)
        idx_set = set(idx)
        candidates = [i for i in range(n) if i not in idx_set]
        add = candidates[:missing]
        idx = np.sort(np.concatenate([idx, add]))
    return x[idx], y[idx], idx


def print_table(headers: list[str], rows: list[list[str]], title: str | None = None) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    if title:
        print(f"\n{title}")

    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(line)
    print(sep)
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main(out_dir: str):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plt = None

    url = build_open_elevation_url(COORDS)
    print("Запит до API:", url)

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "results" not in data:
        raise ValueError(f"API повернув неочікувану відповідь: {data}")

    results = data["results"]
    n = len(results)
    print("Кількість вузлів:", n)
    if n != len(COORDS):
        raise ValueError(f"Очікувалось {len(COORDS)} вузлів, отримано {n}")

    lats = np.array([p["latitude"] for p in results], dtype=float)
    lons = np.array([p["longitude"] for p in results], dtype=float)
    elev = np.array([p["elevation"] for p in results], dtype=float)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    nodes_rows = [[f"{i}", f"{lats[i]:.6f}", f"{lons[i]:.6f}", f"{elev[i]:.2f}"] for i in range(n)]
    with open(Path(out_dir) / "tabulation_nodes.txt", "w", encoding="utf-8") as f:
        f.write("№\tLatitude\tLongitude\tElevation(m)\n")
        for row in nodes_rows:
            f.write("\t".join(row) + "\n")
    print_table(["№", "Latitude", "Longitude", "Elevation(m)"], nodes_rows, "Табуляція вузлів")

    dist = cumulative_distances(list(zip(lats, lons)))
    dist_rows = [[f"{i}", f"{dist[i]:.2f}", f"{elev[i]:.2f}"] for i in range(n)]
    with open(Path(out_dir) / "tabulation_distance_elevation.txt", "w", encoding="utf-8") as f:
        f.write("№\tDistance(m)\tElevation(m)\n")
        for row in dist_rows:
            f.write("\t".join(row) + "\n")
    print_table(["№", "Distance(m)", "Elevation(m)"], dist_rows, "Табуляція відстань-висота")

    spline_full = NaturalCubicSpline(dist, elev)
    xx = np.linspace(dist[0], dist[-1], 2000)
    yy_full = spline_full.eval(xx)

    spline_results: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for k in (10, 15, 20):
        xk, yk, _ = pick_evenly(dist, elev, k)
        spline_k = NaturalCubicSpline(xk, yk)
        yy_k = spline_k.eval(xx)
        spline_results[k] = (xk, yk, yy_k)

        if plt is None:
            continue

        plt.figure()
        plt.plot(dist, elev, marker="o", linestyle="None", label="Дискретні вузли (усі)")
        plt.plot(xk, yk, marker="o", linestyle="None", label=f"Вузли для сплайна (k={k})")
        plt.plot(xx, yy_k, label=f"Сплайн (k={k})")
        plt.plot(xx, yy_full, label="Еталон (сплайн по всіх вузлах)")
        plt.legend()
        plt.grid(True)
        plt.xlabel("Кумулятивна відстань, м")
        plt.ylabel("Висота, м")
        plt.title(f"Профіль висоти (k={k})")
        plt.tight_layout()

    if plt is not None:
        plt.figure()
        plt.plot(xx, yy_full, linewidth=2, label="Еталон (усі вузли)")
        for k in (10, 15, 20):
            xk, yk, yy_k = spline_results[k]
            plt.plot(xx, yy_k, label=f"Сплайн (k={k})")
            plt.scatter(xk, yk, s=12, alpha=0.6)
        plt.grid(True)
        plt.xlabel("Кумулятивна відстань, м")
        plt.ylabel("Висота, м")
        plt.title("Накладання трьох сплайнів (k=10,15,20)")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="output", help="Куди зберігати табличні файли")
    args = parser.parse_args()

    main(out_dir=args.out_dir)
