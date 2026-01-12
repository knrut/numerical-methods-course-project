# part_2/h_interpolation.py
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("TkAgg")
from src.part_2.h_data import dT_data, h_data

def lagrange_interpolant(x_nodes: np.ndarray, y_nodes: np.ndarray):
    """
    Zwraca funkcję P(x), która interpoluje dane metodą Lagrange’a.
    x_nodes, y_nodes – węzły interpolacji (1D numpy arrays)
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    n = len(x_nodes)

    def P(x):
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)

        for i in range(n):
            # L_i(x)
            Li = np.ones_like(x)
            xi = x_nodes[i]
            for j in range(n):
                if i != j:
                    Li *= (x - x_nodes[j]) / (xi - x_nodes[j])
            result += y_nodes[i] * Li

        return result

    return P


if __name__ == "__main__":
    # import danych z projektu
    from src.part_2.h_data import dT_data, h_data

    # Tworzymy interpolant Lagrange’a
    P = lagrange_interpolant(dT_data, h_data)

    # Siatka do wykresu
    xx = np.linspace(dT_data.min(), dT_data.max(), 400)
    yy = P(xx)

    # Wykres
    plt.figure(figsize=(8, 5))
    plt.plot(dT_data, h_data, "or", label="Dane pomiarowe")
    plt.plot(xx, yy, "-b", label="Interpolacja Lagrange’a")
    plt.xlabel("ΔT")
    plt.ylabel("h(ΔT)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()