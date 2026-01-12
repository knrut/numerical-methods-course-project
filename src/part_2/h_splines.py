import numpy as np
import matplotlib
matplotlib.use("TkAgg")


def phi(xi: float, h: float, x: float | np.ndarray) -> np.ndarray:
    """
    Funkcja bazowa splajnu kubicznego (B-splajn) jak w materiałach.
    xi – środek funkcji bazowej
    h  – krok siatki
    """
    x = np.asarray(x, dtype=float)
    t = np.abs(x - xi) / h
    out = np.zeros_like(x, dtype=float)

    mask1 = t < 1
    out[mask1] = (4 - 6 * t[mask1] ** 2 + 3 * t[mask1] ** 3)

    mask2 = (t >= 1) & (t < 2)
    out[mask2] = (2 - t[mask2]) ** 3

    return out


def build_spline_on_phi_basis(xk: np.ndarray, yk: np.ndarray):
    """
    Splajn kubiczny na bazie funkcji φ_i
    Zakładamy równoodległe węzły xk.
    Zwraca funkcję S3(x).
    """
    xk = np.asarray(xk, dtype=float)
    yk = np.asarray(yk, dtype=float)

    n = len(xk) - 1            # w materiałach: n = liczba przedziałów
    h = xk[1] - xk[0]
    a = xk[0]
    b = xk[-1]

    # Przybliżenia pochodnych brzegowych (zamiast df(a), df(b))
    alpha = (yk[1] - yk[0]) / h
    beta = (yk[-1] - yk[-2]) / h

    # Układ równań jak w materiałach: A * c_inner = rhs
    # c ma długość n+3 dla indeksów i = -1..n+1
    A = np.zeros((n + 1, n + 1), dtype=float)
    rhs = np.zeros(n + 1, dtype=float)

    # Pierwszy wiersz
    A[0, 0] = 4
    if n >= 1:
        A[0, 1] = 2
    rhs[0] = yk[0] + (h / 3) * alpha

    # Środek
    for i in range(1, n):
        A[i, i - 1] = 1
        A[i, i] = 4
        A[i, i + 1] = 1
        rhs[i] = yk[i]

    # Ostatni wiersz
    A[n, n] = 4
    if n >= 1:
        A[n, n - 1] = 2
    rhs[n] = yk[n] - (h / 3) * beta

    # Rozwiązanie współczynników: odpowiadają c_0..c_n (w sensie materiałów)
    c = np.zeros(n + 3, dtype=float)     # indeksy: 0..n+2 => i=-1..n+1
    c[1:n + 2] = np.linalg.solve(A, rhs) # c[1]..c[n+1]

    # c_{-1} i c_{n+1}
    c[0] = c[2] - (h / 3) * alpha
    c[n + 2] = c[n] + (h / 3) * beta

    def S3(x_eval: float | np.ndarray) -> np.ndarray:
        x_eval = np.asarray(x_eval, dtype=float)
        s = np.zeros_like(x_eval, dtype=float)

        # suma po i=-1..n+1
        for i in range(-1, n + 2):
            xi = a + i * h
            s += c[i + 1] * phi(xi, h, x_eval)

        return s

    return S3


def make_spline_approx(dT: np.ndarray, h_data: np.ndarray, num_points: int = 20):
    """
    Jak w wymaganiach: najpierw robimy węzły równoległe, potem splajn na bazie φ.
    """
    dT = np.asarray(dT, dtype=float)
    h_data = np.asarray(h_data, dtype=float)

    # węzły równoległe
    xk = np.linspace(dT.min(), dT.max(), num_points)
    yk = np.interp(xk, dT, h_data)

    return build_spline_on_phi_basis(xk, yk)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.part_2.h_data import dT_data, h_data

    spline = make_spline_approx(dT_data, h_data, num_points=20)

    xx = np.linspace(dT_data.min(), dT_data.max(), 400)
    yy = spline(xx)

    plt.figure(figsize=(8, 5))
    plt.plot(dT_data, h_data, "or", label="Dane pomiarowe")
    plt.plot(xx, yy, "-m", label="Splajn kubiczny na bazie φ")
    plt.xlabel("ΔT")
    plt.ylabel("h(ΔT)")
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()
