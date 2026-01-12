import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from src.part_1.params import Params
from src.part_1.euler import euler_explicit, euler_modified

from src.part_2.h_data import dT_data, h_data
from src.part_2.h_interpolation import lagrange_interpolant
from src.part_2.h_least_squares import make_poly_least_squares
from src.part_2.h_splines import make_spline_approx


# =========================
# RHS z h(ΔT)
# =========================
def heat_exchange_rhs_variable_h(t: float, T: np.ndarray, p: Params, h_func):
    Tb, Tw = T
    delta_T = Tb - Tw
    h = float(h_func(delta_T))

    dTb_dt = (h * p.area / (p.rod_mass * p.rod_specific_heat)) * (Tw - Tb)
    dTw_dt = (h * p.area / (p.fluid_mass * p.fluid_specific_heat)) * (Tb - Tw)

    return np.array([dTb_dt, dTw_dt])


def run_simulation(dt: float, h_method: str):
    # --- wybór funkcji h(ΔT) ---
    if h_method == "lagrange":
        h_func = lagrange_interpolant(dT_data, h_data)
        label = "Lagrange"
    elif h_method == "mnk":
        m = 5
        h_func, _ = make_poly_least_squares(dT_data, h_data, m=m)
        label = f"MNK (m={m})"
    elif h_method == "spline":
        h_func = make_spline_approx(dT_data, h_data, num_points=20)
        label = "Splajn kubiczny"
    else:
        raise ValueError("Nieznana metoda h(ΔT)")

    # --- parametry fizyczne ---
    params = Params(
        area=0.0109,
        rod_mass=0.2,
        fluid_mass=2.5,
        rod_specific_heat=3.85,
        fluid_specific_heat=4.1813,
        heat_transfer_coeff=170.0,  # nieużywane
    )

    rhs = lambda t, T, p: heat_exchange_rhs_variable_h(t, T, p, h_func)

    T0 = np.array([1200.0, 25.0])
    t_end = 5.0

    t_euler, T_euler = euler_explicit(rhs, T0, dt, t_end, params)
    t_heun, T_heun = euler_modified(rhs, T0, dt, t_end, params)

    # --- temperatury końcowe (DO RAPORTU) ---
    Tb_e, Tw_e = T_euler[-1]
    Tb_h, Tw_h = T_heun[-1]

    print(f"\nMetoda h(ΔT): {label}")
    print(f"Euler : Tb_end = {Tb_e:.3f} °C, Tw_end = {Tw_e:.3f} °C")
    print(f"Heun  : Tb_end = {Tb_h:.3f} °C, Tw_end = {Tw_h:.3f} °C")

    # --- wykres ---
    plt.figure(figsize=(8, 5))
    plt.plot(t_euler, T_euler[:, 0], label="Tb – Euler")
    plt.plot(t_euler, T_euler[:, 1], label="Tw – Euler")
    plt.plot(t_heun, T_heun[:, 0], "--", label="Tb – Heun")
    plt.plot(t_heun, T_heun[:, 1], "--", label="Tw – Heun")

    plt.title(f"Chłodzenie pręta – h(ΔT): {label}")
    plt.xlabel("t [s]")
    plt.ylabel("Temperatura [°C]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dt = 0.001

    for method in ["lagrange", "mnk", "spline"]:
        run_simulation(dt=dt, h_method=method)
