# velocity.py

import numpy as np
import MDAnalysis as mda
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from numba import njit

from .spectrum import * 


@njit
def jit_decompose_velocity_core(positions, velocities, masses, is_linear):
    """
    JIT-accelerated core function to decompose atomic velocities into translational, rotational, and vibrational components.

    Parameters
    ----------
    positions : np.ndarray
        Atomic positions, shape (n_atoms, 3)
    velocities : np.ndarray
        Atomic velocities, shape (n_atoms, 3)
    masses : np.ndarray
        Atomic masses, shape (n_atoms,)
    is_linear : bool
        Whether the molecule is linear (affects principal moments)

    Returns
    -------
    com_vel : np.ndarray
        Translational center-of-mass velocity, shape (3,)
    omega : np.ndarray
        Angular velocity vector, shape (3,)
    vib_vel : np.ndarray
        Vibrational velocities, shape (n_atoms, 3)
    evl : np.ndarray
        Principal moments of inertia, shape (3,)
    """
    # Calculate the center of mass position and velocity
    n_atoms = positions.shape[0]
    com_pos = np.sum(masses[:, None] * positions, axis=0) / np.sum(masses)
    com_vel = np.sum(masses[:, None] * velocities, axis=0) / np.sum(masses)
    rel_pos = positions - com_pos
    rel_vel = velocities - com_vel
    
    # Calculate angular momentum and inertia tensor
    ang_mom = np.sum(masses[:, None] * np.cross(rel_pos, rel_vel), axis=0)
    inertia = np.zeros((3, 3))
    for i in range(n_atoms):
        m = masses[i]
        x, y, z = rel_pos[i]
        # Diagonal elements
        inertia[0, 0] += m * (y**2 + z**2)  # I_xx
        inertia[1, 1] += m * (x**2 + z**2)  # I_yy
        inertia[2, 2] += m * (x**2 + y**2)  # I_zz
        # Off-diagonal elements (note the minus sign)
        inertia[0, 1] -= m * x * y          # I_xy
        inertia[1, 0] = inertia[0, 1]
        inertia[0, 2] -= m * x * z          # I_xz
        inertia[2, 0] = inertia[0, 2]
        inertia[1, 2] -= m * y * z          # I_yz
        inertia[2, 1] = inertia[1, 2]
    
    # Diagonalize inertia tensor to get principal moments and axes
    evl, evt = np.linalg.eigh(inertia)
    idx = np.argsort(evl)
    evl = evl[idx]
    evt = evt[:, idx]
    if is_linear and n_atoms > 1:
        evl[0] = 0.0  # For linear molecules, set smallest moment to zero
    
    # Calculate angular velocities along principal axes
    pomega = np.zeros(3)
    for i in range(3):
        if evl[i] > 1e-9:
            L_principal_i = np.dot(ang_mom, evt[:, i].copy())
            pomega[i] = L_principal_i / evl[i]
    omega = np.dot(evt, pomega)
    
    # Calculate rotational and vibrational velocities
    vr = np.cross(omega, rel_pos)
    vib_vel = rel_vel - vr
    
    return com_vel, omega, vib_vel, evl

def _decompose_residue_velocities_framewise(positions, velocities, masses, is_linear, n_frames):
    """
    Decompose velocity time series into components frame by frame.
    
    Parameters
    ----------
    positions : np.ndarray
        Positions array of shape (n_frames, n_atoms, 3)
    velocities : np.ndarray
        Velocities array of shape (n_frames, n_atoms, 3)
    masses : np.ndarray
        Masses array of shape (n_atoms,)
    is_linear : bool
        Whether the molecule is linear
    n_frames : int
        Number of frames to process
        
    Returns
    -------
    tuple
        (vt, vv, omega, evl) arrays with time dimension
    """
    n_atoms = positions.shape[1]
    vt = np.zeros((n_frames, 3))
    omega = np.zeros((n_frames, 3))
    vv = np.zeros((n_frames, n_atoms, 3))
    evl = np.zeros((n_frames, 3))
    
    # Process each frame
    for frame in range(n_frames):
        vt[frame], omega[frame], vv[frame], evl[frame] = jit_decompose_velocity_core(
            positions[frame], velocities[frame], masses, is_linear
        )
    return vt, omega, vv, evl

def _process_residue_wrapper(args):
    """Helper function to unpack arguments for multiprocessing"""
    return _process_residue(*args)

def _process_residue(positions, velocities, masses, is_linear, dt, temperature, FILTERING, filter_window, residue_info):
    """
    Process a single residue to calculate its translational, vibrational and rotational DOS.
    
    Parameters
    ----------
    positions : np.ndarray
        Pre-loaded positions array for this residue (n_frames, n_atoms, 3)
    velocities : np.ndarray
        Pre-loaded velocities array for this residue (n_frames, n_atoms, 3)
    masses : np.ndarray
        Masses array for this residue's atoms
    is_linear : bool
        Whether molecules should be treated as linear
    dt : float
        Time step between frames in ps
    temperature : float
        Temperature in K
    FILTERING : bool
        Whether to apply Blackman window and filtering
    filter_window : int
        Window size for Savitsky-Golay filter
    residue_info : tuple
        (resname, resid) for progress tracking
        
    Returns
    -------
    freqs : numpy.ndarray
        Frequencies at which DOS is evaluated
    tDOS : numpy.ndarray
        Translational density of states
    vDOS : numpy.ndarray
        Vibrational density of states
    rDOS : numpy.ndarray
        Rotational density of states
    """
    resname, resid = residue_info
    n_frames = positions.shape[0]
    molecule_mass = np.sum(masses)
    
    # Decompose velocities frame by frame
    vt, omega, vv, evl = _decompose_residue_velocities_framewise(
        positions, velocities, masses, is_linear, n_frames
    )
    
    # Calculate DOS for each component
    freqs, tDOS = translational_density_of_states(vt, molecule_mass, dt, temperature, FILTERING, filter_window)
    _, vDOS = vibrational_density_of_states(vv, masses, dt, temperature, FILTERING, filter_window)
    
    # For rotational DOS, we need time-averaged principal moments
    I_avg = np.mean(evl, axis=0)  # Average over frames
    _, rDOS = rotational_density_of_states(omega, I_avg, dt, temperature, FILTERING, filter_window)

    return freqs, tDOS, vDOS, rDOS, I_avg


def decompose_trajectory_velocities(residue_data, is_linear=False, n_workers=1, dt=0.001, temperature=300.0, 
                               FILTERING=False, filter_window=5):
    """
    Decompose the velocities of all atoms in a trajectory into translational, rotational, and vibrational components,
    then compute the density of states (DOS) for each component per molecule.
    Uses multiprocessing for parallel processing of residues.
    
    Parameters
    ----------
    residue_data : list
        List of tuples containing pre-loaded trajectory data for each residue.
        Each tuple contains:
        [0] positions: array of shape (n_frames, n_atoms, 3)
        [1] velocities: array of shape (n_frames, n_atoms, 3)
        [2] masses: array of shape (n_atoms,)
        [3] info: tuple of (resname, resid)
    is_linear : bool, optional
        Whether molecules should be treated as linear
    n_workers : int, optional
        Number of parallel workers
    dt : float, optional
        Time step between frames in ps
    temperature : float, optional
        Temperature in K for DOS calculation
    FILTERING : bool, optional
        Whether to apply Blackman window and Savitsky-Golay filtering
    filter_window : int, optional
        Window size for Savitsky-Golay filter
        
    Returns
    -------
    tuple
        (freqs, tDOS_total, vDOS_total, rDOS_total, I_lk) where:
        freqs: frequency array
        tDOS_total: total translational DOS
        vDOS_total: total vibrational DOS
        rDOS_total: total rotational DOS
        I_lk: principal moments of inertia averaged over frames, per residue, shape (n_residues, 3)
    """
    n_residues = len(residue_data)
    
    # Parallelize over residues
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Create list of arguments for each residue - data is already in tuple form
        process_args = [(positions, velocities, masses, is_linear, dt, temperature, FILTERING, filter_window, info)
                       for positions, velocities, masses, info in residue_data]
        for res_dos in tqdm(executor.map(_process_residue_wrapper, process_args), 
                           total=n_residues, desc="DoS calculation"):
            results.append(res_dos)
    
    # Sum up all DOS components
    freqs = results[0][0]  # Frequencies will be the same for all residues
    tDOS_total = np.zeros_like(results[0][1])
    vDOS_total = np.zeros_like(results[0][2])
    rDOS_total = np.zeros_like(results[0][3])
    I_lk = np.zeros((n_residues, results[0][4].shape[0]), dtype=results[0][4].dtype)
    
    for i, (_, tDOS, vDOS, rDOS, I_avg) in enumerate(results):
        tDOS_total += tDOS
        vDOS_total += vDOS
        rDOS_total += rDOS
        I_lk[i] = I_avg
        
    return freqs, tDOS_total, vDOS_total, rDOS_total, I_lk