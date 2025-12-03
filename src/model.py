import numpy as np
from params import Params


def heat_exchange_rhs(t: float, T: np.ndarray, p: Params) -> np.ndarray:
    """
    Right-hand side of the ODE system describing heat exchange
    between a hot metal rod and a cooling fluid (oil).

    Parameters
    ----------
    t : float
        Current time (unused, included for solver compatibility).
    T : np.ndarray
        State vector:
        T[0] = Tb : rod temperature
        T[1] = Tw : fluid temperature
    p : Params
        Physical parameters defining the system.

    Returns
    -------
    np.ndarray
        Time derivatives [dTb/dt, dTw/dt].
    """

    Tb, Tw = T  # unpack state vector

    # Heat transfer differential equations:
    dTb_dt = (p.heat_transfer_coeff * p.area
              / (p.rod_mass * p.rod_specific_heat)) * (Tw - Tb)

    dTw_dt = (p.heat_transfer_coeff * p.area
              / (p.fluid_mass * p.fluid_specific_heat)) * (Tb - Tw)

    return np.array([dTb_dt, dTw_dt])
