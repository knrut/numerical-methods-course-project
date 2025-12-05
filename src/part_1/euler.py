import numpy as np
from typing import Callable
from params import Params


def euler_explicit(
    rhs: Callable[[float, np.ndarray, Params], np.ndarray],
    T0: np.ndarray,
    dt: float,
    t_end: float,
    params: Params
):
    """
    Explicit (forward) Euler method for solving ODE systems.

    Parameters
    ----------
    rhs : function
        Function computing the right-hand side of the ODE system.
    T0 : np.ndarray
        Initial state vector.
    dt : float
        Time step.
    t_end : float
        End of the simulation time interval.
    params : Params
        Physical system parameters.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Arrays of times and corresponding solution states.
    """
    t = 0.0
    T = np.array(T0, dtype=float)

    times = [t]
    states = [T.copy()]

    while t < t_end - 1e-12:
        dT = rhs(t, T, params)
        T = T + dt * dT
        t = t + dt

        times.append(t)
        states.append(T.copy())

    return np.array(times), np.vstack(states)


def euler_modified(
    rhs: Callable[[float, np.ndarray, Params], np.ndarray],
    T0: np.ndarray,
    dt: float,
    t_end: float,
    params: Params
):
    """
    Modified Euler (Heun's method, RK2) for solving ODE systems.
    This method provides significantly improved accuracy over
    the explicit Euler method, especially for stiff or fast-changing dynamics.
    """
    t = 0.0
    T = np.array(T0, dtype=float)

    times = [t]
    states = [T.copy()]

    while t < t_end - 1e-12:
        k1 = rhs(t, T, params)
        k2 = rhs(t + dt, T + dt * k1, params)

        T = T + dt * 0.5 * (k1 + k2)
        t = t + dt

        times.append(t)
        states.append(T.copy())

    return np.array(times), np.vstack(states)
