import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from src.part_2.h_data import dT_data, h_data

def poly_least_squares_coeffs(x: np.ndarray, y: np.ndarray, m: int) -> np.ndarray:
    """
    Aproksymacja wielomianowa metodą najmniejszych kwadratów.
    Szukamy P_m(x) = a0 + a1 x + ... + am x^m,
    minimalizując sumę kwadratów |M a - y|^2.

    Zwraca wektor współczynników a.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    # Macierz M: kolumny [1, x, x^2, ..., x^m]
    M = np.zeros((n, m + 1))
    M[:, 0] = 1.0
    for j in range(1, m + 1):
        M[:, j] = x**j

    # Normal equations: (M^T M) a = M^T y
    MTM = M.T @ M
    MTy = M.T @ y
    a = np.linalg.solve(MTM, MTy)

    return a


def make_poly_least_squares(x: np.ndarray, y: np.ndarray, m: int):
    """
    Zwraca:
      - funkcję P_m(delta_T)
      - błąd średniokwadratowy bl
    """
    a = poly_least_squares_coeffs(x, y, m)

    def P(delta_T):
        delta_T = np.asarray(delta_T, dtype=float)
        result = np.zeros_like(delta_T, dtype=float)
        for c in reversed(a):
            result = result * delta_T + c
        return result

    # liczymy błąd średniokwadratowy na węzłach pomiarowych
    y_approx = P(x)
    bl = np.sqrt(np.mean((y - y_approx) ** 2))

    return P, bl


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # przykład: stopień m = 3 (dobierasz eksperymentalnie)
    m = 3
    Pm, bl = make_poly_least_squares(dT_data, h_data, m=m)
    print(f"Średni błąd bl dla m = {m}: {bl:.6f}")

    xx = np.linspace(dT_data.min(), dT_data.max(), 400)
    yy = Pm(xx)

    plt.figure(figsize=(8, 5))
    plt.plot(dT_data, h_data, "or", label="Dane pomiarowe")
    plt.plot(xx, yy, "-g", label=f"MNK – wielomian stopnia m = {m}")
    plt.xlabel("ΔT")
    plt.ylabel("h(ΔT)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
