# fluidicity.py

import numpy as np
from scipy.optimize import root_scalar

from .constants import *


def calculate_delta(T: float, V: float, N: float, m: float, s0: float) -> float:
    """
    Calculate the dimensionless Delta parameter used in the 2PT method.
    
    Parameters
    ----------
    T : float
        Temperature in Kelvin
    V : float
        Volume in Å³
    N : float
        Number of molecules
    m : float
        Mass in amu
    s0 : float
        Zero-frequency value of the density of states in ps
        
    Returns
    -------
    float
        Dimensionless Delta parameter
    """
    delta = (2*s0)/(9*N) * (pi*kB*T/m)**(1/2) * (N/V)**(1/3) * (6/pi)**(2/3)
    # Convert units: (ps * sqrt(eV / amu) / Å)
    conv2 = ps * np.sqrt(eV / amu) / angstrom
    return delta*conv2


def _equation_of_fluidicity(f: float, delta: float) -> float:
    """
    The equation that defines the fluidicity parameter.
    
    The fluidicity equation is:
    2δ^(-9/2)f^(15/2) - 6δ^(-3)f^5 - δ^(-3/2)f^(7/2) + 6δ^(-3/2)f^(5/2) + 2f - 2 = 0 ... i.e., g_δ(f) = 0
    for really small δ, this expression is numerically unstable. To stabilize, multiply both sides by δ^(4.5):
    2(f^7.5) - 6(f^5)(δ^1.5) - (δ^3)(f^3.5) + 6(δ^3)(f^2.5) + 2(δ^4.5)f - 2(δ^4.5) = 0
    
    This function implements the stabilized form above, which is algebraically
    equivalent but avoids explicit negative powers of δ.
    
    Parameters
    ----------
    f : float
        Fluidicity parameter (between 0 and 1)
    delta : float
        Delta parameter from calculate_delta()
        
    Returns
    -------
    float
        Value of the fluidicity equation
    """
    # Use numpy.power to support both scalar and array inputs
    f = np.asarray(f)
    delta = float(delta)

    delta_1p5 = delta**1.5   # δ^(3/2)
    delta_3 = delta**3.0     # δ^3
    delta_4p5 = delta**4.5   # δ^(9/2)

    term1 = 2.0 * np.power(f, 7.5)
    term2 = -6.0 * np.power(f, 5.0) * delta_1p5
    term3 = -delta_3 * np.power(f, 3.5)
    term4 = 6.0 * delta_3 * np.power(f, 2.5)
    term5 = 2.0 * delta_4p5 * f - 2.0 * delta_4p5

    result = term1 + term2 + term3 + term4 + term5
    # If the input was scalar, return a Python float for compatibility with root_scalar
    return float(result) if result.shape == () else result


def calculate_fluidicity(delta: float) -> float:
    """
    Calculate the fluidicity parameter using a polynomial root approach.
    
    This function rewrites the stabilized fluidicity equation as a polynomial
    in x = \sqrt{f}, solves for all roots of that polynomial, and then
    selects the physically meaningful solution with [0,1].
    For non-positive Δ, the function returns 0.0.
    
    Parameters
    ----------
    delta : float
        Delta parameter from `calculate_delta`.
        
    Returns
    -------
    float
        Fluidicity parameter f constrained to the interval [0, 1].
    """
    if delta <= 0:
        return 0.0
    
    # Coefficients for the polynomial in terms of x = f^(1/2)
    # The equation: 2x^15 - 6Δ^1.5 x^10 - Δ^3 x^7 + 6Δ^3 x^5 + 2Δ^4.5 x^2 - 2Δ^4.5 = 0
    d15 = delta**1.5
    d30 = delta**3
    d45 = delta**4.5
    
    # Define polynomial coefficients from highest power (x^15) to lowest (x^0)
    coeffs = np.zeros(16)
    coeffs[0]  = 2.0                   # x^15
    coeffs[5]  = -6.0 * d15            # x^10
    coeffs[8]  = -1.0 * d30            # x^7
    coeffs[10] = 6.0 * d30             # x^5
    coeffs[13] = 2.0 * d45             # x^2
    coeffs[15] = -2.0 * d45            # x^0 (constant)
    
    # Find all roots
    roots = np.roots(coeffs)
    
    # We only care about real roots between 0 and 1
    real_roots = roots[np.isreal(roots)].real
    valid_roots = real_roots[(real_roots >= 0) & (real_roots <= 1)]
    
    if len(valid_roots) == 0:
        # Fallback for extreme solid limit
        return 0.0
        
    # Usually there is only one valid root in [0, 1]; x^2 = f
    return np.max(valid_roots)**2

def calculate_fluidicity_old(delta: float, method: str = 'brentq') -> float:
    """
    Calculate the fluidicity parameter by solving the fluidicity equation.
    
    Parameters
    ----------
    delta : float
        Delta parameter from calculate_delta()
        
    Returns
    -------
    float
        Fluidicity parameter (between 0 and 1)
    """
    print(f"INFO: Using {method} method to solve for fluidicity")
    return root_scalar(lambda f: _equation_of_fluidicity(f, delta), 
                      method=method, bracket=[1e-3, 1]).root