import numpy as np
from tqdm import tqdm
from scipy.signal import savgol_filter

from .constants import *

def density_of_states(vel, masses, dt, temperature=300, debug=False):
    """
    Compute the mass-weighted power spectrum of a velocity array.
    vel: (n_frames, n_atoms, 3)
    masses: (n_atoms,)
    dt: time step between frames
    temperature: temperature in K (if provided, multiplies by 2/(kB*T))
    Returns: freqs, dos_sum
    If debug=True, also returns intermediate arrays for inspection.
    """
    conv1 = angstrom**2 * amu / eV / ps**2
    n_frames, n_atoms, n_dim = vel.shape
    n_freqs = n_frames // 2 + 1  # for rfft
    fft_normalization = dt / n_frames
    dos_sum = np.zeros(n_freqs)
    window = np.hanning(n_frames)
    print("INFO: Using Hann windowing before FFT")
    debug_dict = {'fft_result': [], 'power_raw': [], 'power_norm': []} if debug else None
    for j in tqdm(range(n_atoms), desc="Processing atoms", leave=False):
        for k in range(n_dim):
            windowed_signal = vel[:, j, k] * window
            fft_result = np.fft.rfft(windowed_signal)
            power_raw = np.abs(fft_result)**2
            power_norm = power_raw*fft_normalization/np.mean(window**2)
            dos_sum += power_norm * masses[j]
            if debug:
                debug_dict['fft_result'].append(fft_result)
                debug_dict['power_raw'].append(power_raw)
                debug_dict['power_norm'].append(power_norm)
    
    freqs = np.fft.rfftfreq(n_frames, d=dt)
    dos_sum = savgol_filter(dos_sum, window_length=5, polyorder=3)
    if debug:
        debug_dict['dos_sum'] = dos_sum
        debug_dict['freqs'] = freqs
        return debug_dict
    dos_sum = dos_sum * 2/(kB*temperature) * conv1
    return freqs, dos_sum

def rotational_density_of_states(omega_all, I_lk, dt, temperature=300, debug=False):
    """
    Compute the rotational density of states rDOS_sum(nu) from angular velocities and principal moments of inertia.
    omega_all: (n_frames, n_molecules, 3) - angular velocities for each molecule and frame
    I_lk: (n_molecules, 3) - principal moments of inertia for each molecule
    dt: time step between frames
    temperature: temperature in K
    Returns: freqs, rDOS_sum
    If debug=True, also returns intermediate arrays for inspection.
    """
    conv1 = angstrom**2 * amu / eV / ps**2
    n_frames, n_molecules, n_dim = omega_all.shape
    n_freqs = n_frames // 2 + 1  # for rfft
    fft_normalization = dt / n_frames
    window = np.hanning(n_frames)
    print("INFO: Using Hann windowing before FFT")
    rDOS_sum = np.zeros(n_freqs)
    debug_dict = {'fft_result': [], 'power_raw': [], 'power_norm': []} if debug else None
    for j in tqdm(range(n_molecules), desc="Processing molecules", leave=False):
        for k in range(n_dim):
            omega_k = omega_all[:, j, k]
            windowed_signal = omega_k * window
            fft_result = np.fft.rfft(windowed_signal)
            power_raw = np.abs(fft_result) ** 2
            power_norm = power_raw * fft_normalization/np.mean(window**2)
            rDOS_sum += power_norm * I_lk[k]
            if debug:
                debug_dict['fft_result'].append(fft_result)
                debug_dict['power_raw'].append(power_raw)
                debug_dict['power_norm'].append(power_norm)
    
    freqs = np.fft.rfftfreq(n_frames, d=dt)
    rDOS_sum = savgol_filter(rDOS_sum, window_length=5, polyorder=3)
    if debug:
        debug_dict['rDOS_sum'] = rDOS_sum
        debug_dict['freqs'] = freqs
        return debug_dict
    rDOS_sum = rDOS_sum * 2/(kB*temperature) * conv1
    return freqs, rDOS_sum 