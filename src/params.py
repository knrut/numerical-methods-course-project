from dataclasses import dataclass

@dataclass
class Params:
    area: float                 # powierzchnia wymiany ciepła [m^2]
    rod_mass: float            # masa pręta [kg]
    fluid_mass: float          # masa oleju/płynu [kg]
    rod_specific_heat: float   # ciepło właściwe pręta [J/(kg*K)]
    fluid_specific_heat: float # ciepło właściwe płynu [J/(kg*K)]
    heat_transfer_coeff: float # współczynnik przejmowania ciepła h

