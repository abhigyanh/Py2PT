# entropy.py

import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import simpson
from typing import Tuple

from .constants import *
from .output    import *

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
    # Handle invalid input cases
    if f < 0 or f > 1 or delta < 0:
        return float('inf')  # Return infinity for invalid values
    
    # Handle special cases
    if f == 0:
        return -2  # The equation evaluates to -2 when f=0
    # if delta == 0:
    #     return float('inf') if f > 0 else -2  # f must be 0 when delta is 0
    # if f == 1:
    #     return 2  # Force positive value at f=1 to ensure different signs
    
    # The fluidicity equation is:
    # 2δ^(-9/2)f^(15/2) - 6δ^(-3)f^5 - δ^(-3/2)f^(7/2) + 6δ^(-3/2)f^(5/2) + 2f - 2 = 0
    try:
        term1 = 2 * (delta**(-9/2)) * (f**(15/2))
        term2 = -6 * (delta**(-3)) * (f**5)
        term3 = -(delta**(-3/2)) * (f**(7/2))
        term4 = 6 * (delta**(-3/2)) * (f**(5/2))
        term5 = 2 * f - 2
        return term1 + term2 + term3 + term4 + term5
    except (ZeroDivisionError, RuntimeWarning):
        return float('inf')

def calculate_fluidicity(delta: float, method: str = 'bisect') -> float:
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
    method = 'bisect'
    print(f"INFO: Using {method} method to solve for fluidicity")
    return root_scalar(lambda f: _equation_of_fluidicity(f, delta), 
                      method=method, bracket=[1e-6, 1]).root

def decompose_translational_dos(nu: np.ndarray, DOS_tr: np.ndarray, 
                              T: float, V: float, N: float, m: float,
                              save_plot: bool = False) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Decompose translational density of states into gas-like and solid-like components.
    
    Parameters
    ----------
    nu : np.ndarray
        Frequency array
    DOS_tr : np.ndarray
        Translational density of states
    T : float
        Temperature in Kelvin
    V : float
        Volume in Å³
    N : float
        Number of molecules
    m : float
        Mass in amu
    save_plot : bool, optional
        Whether to save a plot of the decomposition, by default True
        
    Returns
    -------
    Tuple[float, float, float, np.ndarray, np.ndarray]
        Delta, fluidicity, s0, gas-like DOS, solid-like DOS
    """
    s0_tr = DOS_tr[0]
    Delta_tr = calculate_delta(T, V, N, m, s0_tr)
    f_tr = calculate_fluidicity(Delta_tr)

    DOS_tr_g = s0_tr/(1 + ((pi*s0_tr*nu)/(6*N*f_tr))**2)
    DOS_tr_s = DOS_tr - DOS_tr_g

    # Ensure DOS_tr_g does not exceed DOS_tr
    # Cap DOS_tr_g at DOS_tr where DOS_tr_g > DOS_tr
    DOS_tr_g = np.where(DOS_tr_g > DOS_tr, DOS_tr, DOS_tr_g)
    # Recalculate DOS_tr_s based on the capped DOS_tr_g
    DOS_tr_s = DOS_tr - DOS_tr_g
    
    # If any negative values in DOS_rot_s, set them zero
    tol = -1e-2
    if np.any(DOS_tr_s < tol):
        print(f"Negative values in DOS_tr_s detected (beyond tolerance = {tol})")
        print(f"s0_tr: {s0_tr}")
        # Print the largest negative value
        most_negative = np.min(DOS_tr_s)
        print(f"Most negative value in DOS_tr_s: {most_negative}")

    if save_plot:
        spectra_list = [
            {'freqs': nu, 'DOS': DOS_tr_g, 'label': 'tr_g'},
            {'freqs': nu, 'DOS': DOS_tr_s, 'label': 'tr_s'},
            {'freqs': nu, 'DOS': DOS_tr, 'label': 'tr'}
        ]
        plot_spectra(spectra_list, filename='DoS_tr.png', xlim=[0,1000])

    return Delta_tr, f_tr, s0_tr, DOS_tr_g, DOS_tr_s

def decompose_rotational_dos(nu: np.ndarray, DOS_rot: np.ndarray,
                           T: float, V: float, N: float, m: float,
                           save_plot: bool = False) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Decompose rotational density of states into gas-like and solid-like components.
    
    Parameters
    ----------
    nu : np.ndarray
        Frequency array
    DOS_rot : np.ndarray
        Rotational density of states
    T : float
        Temperature in Kelvin
    V : float
        Volume in Å³
    N : float
        Number of molecules
    m : float
        Mass in amu
    save_plot : bool, optional
        Whether to save a plot of the decomposition, by default True
        
    Returns
    -------
    Tuple[float, float, float, np.ndarray, np.ndarray]
        Delta, fluidicity, s0, gas-like DOS, solid-like DOS
    """
    DOS_rot = DOS_rot.copy()  # Make a copy to avoid modifying the input

    # First, try with direct value from first point
    s0_rot = DOS_rot[0]
    Delta_rot = calculate_delta(T, V, N, m, s0_rot)
    f_rot = calculate_fluidicity(Delta_rot)
    DOS_rot_g = s0_rot/(1 + ((pi*s0_rot*nu)/(6*N*f_rot))**2)
    DOS_rot_s = DOS_rot - DOS_rot_g

    # If any negative values in DOS_rot_s, do the following compensation
    tol = 1e-2
    if np.any(DOS_rot_s < -tol):
        print(f"- Negative values in DOS_rot_s detected (beyond tolerance = {tol})")
        print(f"s0_rot: {s0_rot}")
        # Print the largest negative value
        most_negative = np.min(DOS_rot_s)
        print(f"Most negative value in DOS_rot_s: {most_negative}")
        # Get the negative indices
        neg_idx = np.where(DOS_rot_g > DOS_rot)
        # Try two-phase decomposition again
        print(f"Compensating by adding extraneous DOS_rot_g to DOS_rot and re-decomposing.")
        DOS_rot[neg_idx] += np.abs(DOS_rot_g[neg_idx] - DOS_rot[neg_idx])
        s0_rot = DOS_rot[0]
        Delta_rot = calculate_delta(T, V, N, m, s0_rot)
        f_rot = calculate_fluidicity(Delta_rot)
        DOS_rot_g = s0_rot/(1 + ((pi*s0_rot*nu)/(6*N*f_rot))**2)
        DOS_rot_s = DOS_rot - DOS_rot_g

    if save_plot:
        spectra_list = [
            {'freqs': nu, 'DOS': DOS_rot_g, 'label': 'rot_g'},
            {'freqs': nu, 'DOS': DOS_rot_s, 'label': 'rot_s'},
            {'freqs': nu, 'DOS': DOS_rot, 'label': 'rot'}
        ]
        plot_spectra(spectra_list, filename='DoS_rot.png', xlim=[0,1000])
    return Delta_rot, f_rot, s0_rot, DOS_rot_g, DOS_rot_s

def decompose_vibrational_dos(nu: np.ndarray, DOS_vib: np.ndarray,
                            save_plot: bool = False) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Decompose vibrational density of states (all solid-like).
    
    Parameters
    ----------
    nu : np.ndarray
        Frequency array
    DOS_vib : np.ndarray
        Vibrational density of states
    save_plot : bool, optional
        Whether to save a plot of the decomposition, by default True
        
    Returns
    -------
    Tuple[float, float, float, np.ndarray, np.ndarray]
        Delta (0), fluidicity (0), s0, gas-like DOS (zeros), solid-like DOS
    """
    s0_vib = DOS_vib[0]
    Delta_vib = 0
    f_vib = 0

    DOS_vib_g = np.zeros_like(nu)
    DOS_vib_s = DOS_vib - DOS_vib_g

    if save_plot:
        spectra_list = [
            {'freqs': nu, 'DOS': DOS_vib_g, 'label': 'vib_g'},
            {'freqs': nu, 'DOS': DOS_vib_s, 'label': 'vib_s'},
            {'freqs': nu, 'DOS': DOS_vib, 'label': 'vib'}
        ]
        plot_spectra(spectra_list, filename='DoS_vib.png', xlim=[0,4000])

    return Delta_vib, f_vib, s0_vib, DOS_vib_g, DOS_vib_s

def calculate_translational_entropy(nu: np.ndarray, DOS_tr_s: np.ndarray, DOS_tr_g: np.ndarray,
                                 f_tr: float, Delta_tr: float, m: float, N: float, V: float,
                                 T: float) -> float:
    """
    Calculate translational entropy using the 2PT method.
    
    Parameters
    ----------
    nu : np.ndarray
        Frequency array (non-zero frequencies only)
    DOS_tr_s : np.ndarray
        Solid-like translational DOS
    DOS_tr_g : np.ndarray
        Gas-like translational DOS
    f_tr : float
        Translational fluidicity
    Delta_tr : float
        Translational Delta parameter
    m : float
        Mass in amu
    N : float
        Number of molecules
    V : float
        Volume in Å³
    T : float
        Temperature in Kelvin
        
    Returns
    -------
    float
        Translational entropy in J/(mol·K)
    """
    kT = kB*T

    # Hard sphere packing fraction (Carnahan-Sterling)
    y = (f_tr**(5/2)) / (Delta_tr**(3/2))
    z = (1 + y + y**2 - y**3) / (1 - y)**3

    # Dimensionless hard sphere entropy
    conv3 = (amu / eV)**(3/2) * angstrom**3 / ps**3
    S_HS = (5/2 + np.log(conv3 * (2*pi*m*kB*T/h**2)**(3/2) * V/N * z/f_tr) + 
            (3*y*y - 4*y)/(1-y)**2)

    # Use a mask to handle zeros or near-zeros
    limit = 1e-6
    bhn = np.where(nu < limit, limit, h*nu/kT)

    # Weighing functions
    W_gas = 1/3 * S_HS
    # For the first term: use the identity or a specialized function
    term1 = bhn / (np.expm1(bhn))
    # For the second term: 
    # To avoid log(0), ensure the argument is slightly above 0
    term2 = -np.log(1 - np.exp(-bhn))
    W_solid = term1 + term2
    # Integrate to get the total entropy
    S_tr = simpson(DOS_tr_g*W_gas, x=nu) + simpson(DOS_tr_s*W_solid, x=nu)
    return S_tr * kB * eVtoJ

def calculate_rotational_entropy(nu: np.ndarray, DOS_rot_s: np.ndarray, DOS_rot_g: np.ndarray,
                              I_1: float, I_2: float, I_3: float, Sigma: float, T: float) -> float:
    """
    Calculate rotational entropy using the 2PT method.
    
    Parameters
    ----------
    nu : np.ndarray
        Frequency array (non-zero frequencies only)
    DOS_rot_s : np.ndarray
        Solid-like rotational DOS
    DOS_rot_g : np.ndarray
        Gas-like rotational DOS
    I_1, I_2, I_3 : float
        Principal moments of inertia in amu·Å²
    Sigma : float
        Rotational symmetry number
    T : float
        Temperature in Kelvin
        
    Returns
    -------
    float
        Rotational entropy in J/(mol·K)
    """
    kT = kB*T

    # Rotational entropy
    conv4 = (eV * ps**2) / (amu * angstrom**2)
    Theta1 = h**2/(8*pi**2*kB*I_1) * conv4
    Theta2 = h**2/(8*pi**2*kB*I_2) * conv4
    Theta3 = h**2/(8*pi**2*kB*I_3) * conv4
    S_R = np.log(pi**(1/2)*e**(3/2)/Sigma * (T**3/(Theta1*Theta2*Theta3))**(1/2))

    # Use a mask to handle zeros or near-zeros
    limit = 1e-6
    bhn = np.where(nu < limit, limit, h*nu/kT)
    # Weighing functions
    # For the first term: use the identity or a specialized function
    term1 = bhn / (np.expm1(bhn))
    # For the second term: 
    # To avoid log(0), ensure the argument is slightly above 0
    term2 = -np.log(1 - np.exp(-bhn))
    W_solid = term1 + term2
    W_gas = 1/3 * S_R

    S_rot = simpson(DOS_rot_g*W_gas, x=nu) + simpson(DOS_rot_s*W_solid, x=nu)
    return S_rot * kB * eVtoJ

def calculate_vibrational_entropy(nu: np.ndarray, DOS_vib: np.ndarray, T: float) -> float:
    """
    Calculate vibrational entropy using the 2PT method.
    
    Parameters
    ----------
    nu : np.ndarray
        Frequency array (non-zero frequencies only)
    DOS_vib : np.ndarray
        Vibrational density of states
    T : float
        Temperature in Kelvin
        
    Returns
    -------
    float
        Vibrational entropy in J/(mol·K)
    """
    kT = kB*T
    # Use a mask to handle zeros or near-zeros
    limit = 1e-6
    bhn = np.where(nu < limit, limit, h*nu/kT)
    # For the first term: use the identity or a specialized function
    term1 = bhn / (np.expm1(bhn))
    # For the second term: 
    # To avoid log(0), ensure the argument is slightly above 0
    term2 = -np.log(1 - np.exp(-bhn))
    W_solid = term1 + term2

    S_vib = simpson(DOS_vib*W_solid, x=nu)
    return S_vib * kB * eVtoJ 
