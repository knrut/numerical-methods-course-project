import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def phi(xi: float, h: float, x: float) -> float:
    """
        Funkcja bazowa splajnu kubicznego (B-splajn).
        xi – środek funkcji bazowej
        h  – krok siatki
        """
    t = abs(x - xi) / h

    if t < 1:
        return (4 - 6 * t ** 2 + 3 * t ** 3) / 6
    elif t < 2:
        return (2 - t) ** 3 / 6
    else:
        return 0.0


def plot_spline_basis(a: float, b: float, num_knots: int = 5):
    xd = np.linspace(a, b, 400)
    xk = np.linspace(a, b, num_knots)
    h = xk[1] - xk[0]

    plt.figure()
    for i in range(-1, num_knots + 1):
        xi = xk[0] + i * h
        yd = [phi(xi, h, x) for x in xd]
        plt.plot(xd, yd, label=f"φ_{i}")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_spline_basis(-5, 5, num_knots=6)


