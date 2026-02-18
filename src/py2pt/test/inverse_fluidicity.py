import numpy as np
from scipy.optimize import root_scalar
from .entropy import calculate_delta, _equation_of_fluidicity

# Example fixed parameters
T = 298.15  # Kelvin
rho = 0.9968 # g/cm3
N = 2000.0   # Number of molecules
m = 18.0   # amu (e.g., water)

# Calculate total mass in grams
mass_amu = N * m
mass_g = mass_amu * 1.66053906660e-24  # 1 amu = 1.66054e-24 g
# Calculate volume in cm^3
V_cm3 = mass_g / rho
# Convert cm^3 to Angstrom^3 (1 cm^3 = 1e24 Angstrom^3)
V = V_cm3 * 1e24

# Specify the target fluidicity value
f_target = 0.29  # Example value

def fluidicity_inverse_problem(s0_guess=1.0):
    """
    Solve for the zero-frequency DOS value s0 that yields the target fluidicity f_target.

    Uses module-level T, V, N, m and f_target. Finds s0 such that the fluidicity equation
    is satisfied when delta = calculate_delta(T, V, N, m, s0). Prints the result or a
    convergence message.

    Parameters
    ----------
    s0_guess : float, optional
        Initial guess for s0 (root_scalar uses bracket [1e-8, 1e8]), by default 1.0
    """
    def func(s0):
        delta = calculate_delta(T, V, N, m, s0)
        return _equation_of_fluidicity(f_target, delta)
    # Use root_scalar to solve for s0
    result = root_scalar(func, bracket=[1e-8, 1e8], method='brentq')
    if result.converged:
        print(f"For fluidicity f = {f_target}, solved s0 = {result.root} ps")
    else:
        print("Root finding did not converge.")

if __name__ == "__main__":
    fluidicity_inverse_problem() 