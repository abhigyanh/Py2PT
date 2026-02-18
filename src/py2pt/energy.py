# energy.py

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

from .constants import h, kB, eVtoJ


def _bhn(nu: np.ndarray, T: float, *, limit: float = 1e-6) -> np.ndarray:
    """
    Compute bhn = h*nu/(kB*T) with a small-frequency floor for stability.

    Parameters
    ----------
    nu : np.ndarray
        Frequency array (ps^-1 in Py2PT's convention).
    T : float
        Temperature in Kelvin.
    limit : float, optional
        Minimum frequency used to avoid division-by-zero at nu=0.

    Returns
    -------
    np.ndarray
        Dimensionless bhn array.
    """
    kT = kB * T
    nu_eff = np.where(nu < limit, limit, nu)
    return h * nu_eff / kT


def _W_solid(bhn: np.ndarray) -> np.ndarray:
    """
    Solid-like (quantum harmonic) energy weight in units of kT.

    W_solid = bhn/2 + bhn/(exp(bhn)-1)
    """
    return 0.5 * bhn + bhn / np.expm1(bhn)


def calculate_translational_energy(
    nu: np.ndarray,
    DOS_tr_s: np.ndarray,
    DOS_tr_g: np.ndarray,
    T: float,
) -> float:
    """
    Calculate translational internal energy using the 2PT method.

    Parameters
    ----------
    nu : np.ndarray
        Frequency array (ps^-1).
    DOS_tr_s : np.ndarray
        Solid-like translational DOS.
    DOS_tr_g : np.ndarray
        Gas-like translational DOS.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Translational energy in J/mol.
    """
    bhn = _bhn(nu, T)
    W_solid_trn = _W_solid(bhn)
    W_gas_trn = 0.5

    E_over_kT = simpson(DOS_tr_g * W_gas_trn, x=nu) + simpson(DOS_tr_s * W_solid_trn, x=nu)
    return E_over_kT * (kB * T) * eVtoJ


def calculate_rotational_energy(
    nu: np.ndarray,
    DOS_rot_s: np.ndarray,
    DOS_rot_g: np.ndarray,
    T: float,
) -> float:
    """
    Calculate rotational internal energy using the 2PT method.

    Parameters
    ----------
    nu : np.ndarray
        Frequency array (ps^-1).
    DOS_rot_s : np.ndarray
        Solid-like rotational DOS.
    DOS_rot_g : np.ndarray
        Gas-like rotational DOS.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Rotational energy in J/mol.
    """
    bhn = _bhn(nu, T)
    W_solid_rot = _W_solid(bhn)
    W_gas_rot = 0.5

    E_over_kT = simpson(DOS_rot_g * W_gas_rot, x=nu) + simpson(DOS_rot_s * W_solid_rot, x=nu)
    return E_over_kT * (kB * T) * eVtoJ


def calculate_vibrational_energy(nu: np.ndarray, DOS_vib: np.ndarray, T: float) -> float:
    """
    Calculate vibrational internal energy using the 2PT method.

    Parameters
    ----------
    nu : np.ndarray
        Frequency array (ps^-1).
    DOS_vib : np.ndarray
        Vibrational DOS (treated as fully solid-like in 2PT).
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Vibrational energy in J/mol.
    """
    bhn = _bhn(nu, T)
    W_solid_vib = _W_solid(bhn)

    E_over_kT = simpson(DOS_vib * W_solid_vib, x=nu)
    return E_over_kT * (kB * T) * eVtoJ

def calculate_zero_point_energy(nu: np.ndarray, DOS_vib: np.ndarray, T: float) -> float:
    """
    Calculate zero-point energy using the 2PT method.
    Reference: A. Tiwari, C. Honingh, B. Ensing; Accurate calculation of zero point energy from molecular dynamics simulations 
    of liquids and their mixtures. J. Chem. Phys. 28 December 2019; 151 (24): 244124. https://doi.org/10.1063/1.5131145

    Parameters
    ----------
    nu : np.ndarray
        Frequency array (ps^-1).
    DOS_vib : np.ndarray
        Vibrational DOS (treated as fully solid-like in 2PT).
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Zero-point energy in J/mol.
    """
    bhn = _bhn(nu, T)
    W_zpe = bhn*kB*T/2  # hnu/2
    return simpson(DOS_vib * W_zpe, x=nu) * eVtoJ