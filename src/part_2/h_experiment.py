import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from src.part_2.h_data import dT_data, h_data
from src.part_2.h_interpolation import lagrange_interpolant
from src.part_2.h_least_squares import make_poly_least_squares
from src.part_2.h_splines import make_spline_approx


EPS = 1e-12  # żeby nie dzielić przez zero przy błędzie względnym


def compute_error_metrics(y_true: np.ndarray, y_approx: np.ndarray) -> dict:
    """
    Liczy podstawowe miary błędu:
    - średni błąd bezwzględny (MAE)
    - błąd RMS (RMSE)
    - błąd maksymalny
    - średni i maksymalny błąd względny
    """
    y_true = np.asarray(y_true, dtype=float)
    y_approx = np.asarray(y_approx, dtype=float)

    err = y_approx - y_true
    abs_err = np.abs(err)

    mae = np.mean(abs_err)
    rmse = np.sqrt(np.mean(err**2))
    max_abs = np.max(abs_err)

    rel_err = abs_err / np.maximum(np.abs(y_true), EPS)
    mean_rel = np.mean(rel_err)
    max_rel = np.max(rel_err)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAX_ABS": max_abs,
        "MEAN_REL": mean_rel,
        "MAX_REL": max_rel,
    }


def print_metrics(name: str, metrics: dict):
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        print(f"{k:8s} = {v: .6e}")


if __name__ == "__main__":
    x = dT_data
    y = h_data

    # 1) Interpolacja wielomianowa (Lagrange)
    P_interp = lagrange_interpolant(x, y)
    y_interp_at_nodes = P_interp(x)
    metrics_interp = compute_error_metrics(y, y_interp_at_nodes)

    # 2) Aproksymacja wielomianowa MNK dla kilku stopni
    degrees = [1, 2, 3, 4, 5]  # można zmienić wg uznania
    mnk_results = {}
    mnk_functions = {}

    print("Eksperyment MNK (różne stopnie):")
    for m in degrees:
        Pm, bl_mnk = make_poly_least_squares(x, y, m=m)
        mnk_functions[m] = Pm

        y_mnk_at_nodes = Pm(x)
        metrics_m = compute_error_metrics(y, y_mnk_at_nodes)
        metrics_m["BL_MNK"] = bl_mnk  # dla porównania z RMSE
        mnk_results[m] = metrics_m

        print(f"\n--- Stopień m = {m} ---")
        for k, v in metrics_m.items():
            print(f"{k:8s} = {v: .6e}")

    # wybór najlepszego stopnia MNK wg najmniejszego RMSE
    best_m = min(degrees, key=lambda m: mnk_results[m]["RMSE"])
    print(f"\n>>> Najlepszy stopień MNK wg RMSE: m = {best_m}")

    # 3) Splajn kubiczny
    spline = make_spline_approx(x, y, num_points=20)
    y_spline_at_nodes = spline(x)
    metrics_spline = compute_error_metrics(y, y_spline_at_nodes)

    # Wydruk podsumowania
    print_metrics("Interpolacja wielomianowa (Lagrange)", metrics_interp)
    print_metrics(f"MNK – najlepszy stopień m = {best_m}", mnk_results[best_m])
    print_metrics("Splajn kubiczny (naturalny)", metrics_spline)

    # 4) Wykres porównawczy
    xx = np.linspace(x.min(), x.max(), 400)

    yy_interp = P_interp(xx)
    yy_mnk_best = mnk_functions[best_m](xx)
    yy_spline = spline(xx)

    plt.figure(figsize=(9, 6))
    plt.plot(x, y, "or", label="Dane pomiarowe")
    plt.plot(xx, yy_interp, "-b", label="Interpolacja wielomianowa (Lagrange)")
    plt.plot(xx, yy_mnk_best, "-g", label=f"MNK (m = {best_m})")
    plt.plot(xx, yy_spline, "-m", label="Splajn kubiczny (naturalny)")

    # sensownie przybliżamy skalę na osi y, żeby zobaczyć kształt funkcji h(ΔT)
    plt.ylim(155, 185)

    plt.xlabel("ΔT")
    plt.ylabel("h(ΔT)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
