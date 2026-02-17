import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter

from .constants import *

def translational_density_of_states(vt, molecule_mass, dt, temperature, FILTERING=False, filter_window=5):
    """
    Compute the mass-weighted power spectrum of a velocity array.
    
    Parameters
    ----------
    vt : np.ndarray
        Translational (center of mass) velocity array, shape (n_frames, 3)
    mass : np.float
        Molecular masses, float
    dt : float
        Time step between frames in ps
    temperature : float
        Temperature in K
    FILTERING : bool, optional
        Whether to apply Blackman window and Savitsky-Golay filtering
    filter_window : int, optional
        Window size for Savitsky-Golay filter if FILTERING is True
        
    Returns
    -------
    freqs : np.ndarray
        Frequency array
    tdos_sum : np.ndarray
        Density of states array
    """
    conv1 = angstrom**2 * amu / eV / ps**2
    n_frames, n_dim = vt.shape
    n_freqs = n_frames // 2 + 1  # for rfft
    fft_normalization = dt / n_frames
    tdos_sum = np.zeros(n_freqs)
    if FILTERING:
        window = np.blackman(n_frames)
    else:
        window = np.ones(n_frames)
    for k in range(n_dim):
        windowed_signal = vt[:, k] * window
        fft_result = np.fft.rfft(windowed_signal)
        power_raw = np.abs(fft_result)**2
        power_norm = power_raw*fft_normalization/np.mean(window**2)
        tdos_sum += power_norm * molecule_mass
    freqs = np.fft.rfftfreq(n_frames, d=dt)
    if FILTERING:
        tdos_sum = savgol_filter(tdos_sum, window_length=filter_window, polyorder=3)
    tdos_sum = tdos_sum * 2/(kB*temperature) * conv1
    return freqs, tdos_sum


def vibrational_density_of_states(vel, masses, dt, temperature, FILTERING=False, filter_window=5):
    """
    Compute the mass-weighted power spectrum of a velocity array.
    
    Parameters
    ----------
    vel : np.ndarray
        Velocity array, shape (n_frames, n_atoms, 3)
    masses : np.ndarray
        Atom masses, shape (n_atoms,)
    dt : float
        Time step between frames in ps
    temperature : float
        Temperature in K
    FILTERING : bool, optional
        Whether to apply Blackman window and Savitsky-Golay filtering
    filter_window : int, optional
        Window size for Savitsky-Golay filter if FILTERING is True
        
    Returns
    -------
    freqs : np.ndarray
        Frequency array
    vdos_sum : np.ndarray
        Density of states array
    """
    conv1 = angstrom**2 * amu / eV / ps**2
    n_frames, n_atoms, n_dim = vel.shape
    n_freqs = n_frames // 2 + 1  # for rfft
    fft_normalization = dt / n_frames
    vdos_sum = np.zeros(n_freqs)
    if FILTERING:
        window = np.blackman(n_frames)
    else:
        window = np.ones(n_frames)
    for j in range(n_atoms):
        for k in range(n_dim):
            windowed_signal = vel[:, j, k] * window
            fft_result = np.fft.rfft(windowed_signal)
            power_raw = np.abs(fft_result)**2
            power_norm = power_raw*fft_normalization/np.mean(window**2)
            vdos_sum += power_norm * masses[j]
    freqs = np.fft.rfftfreq(n_frames, d=dt)
    if FILTERING:
        vdos_sum = savgol_filter(vdos_sum, window_length=filter_window, polyorder=3)
    vdos_sum = vdos_sum * 2/(kB*temperature) * conv1
    return freqs, vdos_sum

def rotational_density_of_states(omega, I_lk, dt, temperature, FILTERING=False, filter_window=5):
    """
    Compute the rotational density of states rDOS_sum(nu) from angular velocities and principal moments of inertia
    for a single molecule.
    
    Parameters
    ----------
    omega : np.ndarray
        Angular velocities with shape (n_frames, 3) for a single molecule
    I_lk : np.ndarray
        Principal moments of inertia with shape (3,) for a single molecule
    dt : float 
        Time step between frames in ps
    temperature : float
        Temperature in K
    FILTERING : bool, optional
        Whether to apply Blackman window
    ZERO_CORRECTION : bool, optional
        Whether to apply zero correction
    filter_window : int, optional
        Window size for Savitsky-Golay filter
        
    Returns
    -------
    freqs : np.ndarray
        Frequency array
    rDOS_sum : np.ndarray
        Rotational density of states
        
    Raises
    ------
    ValueError
        If omega has wrong shape (must be 2D) or I_lk is not 1D
    """
    conv1 = angstrom**2 * amu / eV / ps**2
    n_frames, n_dim = omega.shape
    
    # Validate input shapes for single molecule
    if omega.ndim != 2:
        raise ValueError("omega must have shape (n_frames, 3) for single molecule")
    if I_lk.ndim != 1:
        raise ValueError("I_lk must have shape (3,) for single molecule")
    if n_dim != 3 or len(I_lk) != 3:
        raise ValueError("Both omega and I_lk must have 3 components")
        
    n_freqs = n_frames // 2 + 1
    fft_normalization = dt / n_frames
    if FILTERING:
        window = np.blackman(n_frames)
    else:
        window = np.ones(n_frames)
        
    rDOS_sum = np.zeros(n_freqs)
    # Process single molecule
    for k in range(n_dim):
        windowed_signal = omega[:, k] * window
        fft_result = np.fft.rfft(windowed_signal)
        power_raw = np.abs(fft_result) ** 2
        power_norm = power_raw * fft_normalization/np.mean(window**2)
        rDOS_sum += power_norm * I_lk[k]
    freqs = np.fft.rfftfreq(n_frames, d=dt)
    if FILTERING:
        rDOS_sum = savgol_filter(rDOS_sum, window_length=filter_window, polyorder=3)
    rDOS_sum = rDOS_sum * 2/(kB*temperature) * conv1
    return freqs, rDOS_sum 