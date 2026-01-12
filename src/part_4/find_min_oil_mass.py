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


def make_h_func(h_method: str = "spline"):
    if h_method == "lagrange":
        return lagrange_interpolant(dT_data, h_data), "Lagrange"
    elif h_method == "mnk":
        m = 5
        f, _ = make_poly_least_squares(dT_data, h_data, m=m)
        return f, f"MNK (m={m})"
    elif h_method == "spline":
        return make_spline_approx(dT_data, h_data, num_points=20), "splajn kubiczny"
    else:
        raise ValueError("h_method must be: 'lagrange', 'mnk', 'spline'")


def simulate_Tb_at_time(
    mw: float,
    t_target: float,
    dt: float,
    h_func,
    method: str = "heun",
):
    """
    Zwraca Tb(t_target) dla zadanej masy oleju mw.
    """
    params = Params(
        area=0.0109,
        rod_mass=0.2,
        fluid_mass=mw,              # tutaj zmieniamy masę oleju
        rod_specific_heat=3.85,
        fluid_specific_heat=4.1813,
        heat_transfer_coeff=170.0,  # ignorowane, bo liczymy h z h_func
    )

    rhs = lambda t, T, p: heat_exchange_rhs_variable_h(t, T, p, h_func)

    T0 = np.array([1200.0, 25.0])
    t_end = t_target

    if method == "euler":
        t, TT = euler_explicit(rhs, T0, dt, t_end, params)
    elif method == "heun":
        t, TT = euler_modified(rhs, T0, dt, t_end, params)
    else:
        raise ValueError("method must be 'euler' or 'heun'")

    Tb = TT[-1, 0]
    return float(Tb)


def newton_find_mw(
    Tb_target: float = 125.0,
    t_target: float = 0.7,
    dt: float = 0.001,
    h_method: str = "spline",
    ode_method: str = "heun",
    mw0: float = 2.5,
    dm: float = 0.01,
    tol_T: float = 1e-3,
    tol_m: float = 1e-6,
    max_iter: int = 30,
):
    """
    Newton-Raphson dla równania:
        F(mw) = Tb(t_target, mw) - Tb_target = 0
    Pochodna liczona numerycznie:
        F'(mw) ~ (F(mw+dm)-F(mw))/dm
    """
    h_func, h_label = make_h_func(h_method)

    def F(mw):
        Tb = simulate_Tb_at_time(mw, t_target, dt, h_func, method=ode_method)
        return Tb - Tb_target

    history = []

    mw = float(mw0)
    for k in range(1, max_iter + 1):
        f0 = F(mw)
        # pochodna numeryczna
        f1 = F(mw + dm)
        dF = (f1 - f0) / dm

        history.append((k, mw, f0, dF))

        # zabezpieczenia
        if not np.isfinite(f0) or not np.isfinite(dF):
            raise RuntimeError("Newton: wartości niefinitywne (NaN/inf). Zmień mw0/dm/dt lub metodę h.")
        if abs(dF) < 1e-12:
            raise RuntimeError("Newton: pochodna ~ 0 (niestabilność). Zmień dm lub punkt startowy mw0.")

        mw_new = mw - f0 / dF

        # masa musi być dodatnia
        if mw_new <= 0:
            mw_new = 0.1  # minimalne sensowne zabezpieczenie

        # kryteria stopu
        if abs(f0) < tol_T:
            break
        if abs(mw_new - mw) < tol_m:
            mw = mw_new
            break

        mw = mw_new

    return mw, h_label, history


if __name__ == "__main__":
    mw_star, h_label, hist = newton_find_mw(
        Tb_target=125.0,
        t_target=0.7,
        dt=0.001,
        h_method="spline",
        ode_method="heun",
        mw0=2.5,
        dm=0.02,
    )

    print(f"\nWynik Newtona:")
    print(f"  h(ΔT): {h_label}")
    print(f"  minimalna masa oleju mw ≈ {mw_star:.6f} kg\n")

    print("Iteracje (k, mw, F(mw)=Tb-125, F'(mw)):")
    for k, mw, f0, dF in hist:
        print(f"{k:2d}  mw={mw: .6f}   F={f0: .6f}   dF={dF: .6f}")
