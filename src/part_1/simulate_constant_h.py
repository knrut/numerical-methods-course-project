import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


from src.part_1.params import Params
from src.part_1.model import heat_exchange_rhs
from src.part_1.euler import euler_explicit, euler_modified


def run_single_simulation(dt: float):
    # Parametry z treści zadania (Część 1)
    params = Params(
        area=0.0109,
        rod_mass=0.2,
        fluid_mass=2.5,
        rod_specific_heat=3.85,
        fluid_specific_heat=4.1813,
        heat_transfer_coeff=160.0,
    )

    # Warunki początkowe: [Tb(0), Tw(0)]
    T0 = np.array([1200.0, 25.0])
    t_end = 5.0  # [s]

    # Euler zwykły
    t_euler, T_euler = euler_explicit(
        rhs=heat_exchange_rhs,
        T0=T0,
        dt=dt,
        t_end=t_end,
        params=params,
    )

    # Euler zmodyfikowany (Heun)
    t_heun, T_heun = euler_modified(
        rhs=heat_exchange_rhs,
        T0=T0,
        dt=dt,
        t_end=t_end,
        params=params,
    )

    # Wykres
    plt.figure(figsize=(8, 5))
    plt.plot(t_euler, T_euler[:, 0], label="Tb – Euler")
    plt.plot(t_euler, T_euler[:, 1], label="Tw – Euler")

    plt.plot(t_heun, T_heun[:, 0], "--", label="Tb – Modified Euler")
    plt.plot(t_heun, T_heun[:, 1], "--", label="Tw – Modified Euler")

    plt.xlabel("t [s]")
    plt.ylabel("Temperature [°C]")
    plt.title(f"Cooling simulation (dt = {dt})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Na start: jeden sensowny krok
    run_single_simulation(dt=0.001)
