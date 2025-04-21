"""
Calculate moments of inertia for molecules in a trajectory.

This module provides functions to calculate principal moments of inertia
for molecules in an MDAnalysis AtomGroup, with statistical analysis
across all molecules and frames.
"""

import numpy as np
from typing import Tuple
from tqdm import tqdm
from numba import njit
from concurrent.futures import ProcessPoolExecutor


@njit
def calculate_molecule_inertia_tensor(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    Calculate the inertia tensor for a single molecule.
    
    Parameters
    ----------
    positions : np.ndarray
        Atom positions relative to center of mass, shape (n_atoms, 3)
    masses : np.ndarray
        Atom masses, shape (n_atoms,)
        
    Returns
    -------
    np.ndarray
        3x3 inertia tensor
    """
    I = np.zeros((3, 3))
    
    for pos, mass in zip(positions, masses):
        # Diagonal elements
        I[0,0] += mass * (pos[1]**2 + pos[2]**2)
        I[1,1] += mass * (pos[0]**2 + pos[2]**2)
        I[2,2] += mass * (pos[0]**2 + pos[1]**2)
        # Off-diagonal elements
        I[0,1] -= mass * pos[0] * pos[1]
        I[1,0] = I[0,1]
        I[0,2] -= mass * pos[0] * pos[2]
        I[2,0] = I[0,2]
        I[1,2] -= mass * pos[1] * pos[2]
        I[2,1] = I[1,2]
    
    return I


def calculate_average_moments(atomgroup) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate average principal moments of inertia across all molecules in an AtomGroup
    for a single frame.
    
    Parameters
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
        AtomGroup containing the molecules to analyze
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - Principal moments (I₁, I₂, I₃) sorted from smallest to largest
        - Standard deviations of the moments
        - Array of all individual molecule moments, shape (n_molecules, 3)
    """
    n_molecules = len(atomgroup.residues)
    I_tensor = np.zeros((n_molecules, 3))  # Store principal moments for each molecule
    
    for i, res in enumerate(atomgroup.residues):
        # Calculate positions relative to center of mass
        res_pos = res.atoms.positions - res.atoms.center_of_mass()
        res_masses = res.atoms.masses
        
        # Calculate inertia tensor and its eigenvalues
        I = calculate_molecule_inertia_tensor(res_pos, res_masses)
        principal_moments = np.sort(np.linalg.eigvalsh(I))  # Sort from smallest to largest
        I_tensor[i] = principal_moments

    # Calculate statistics
    I_avg = np.mean(I_tensor, axis=0)
    I_std = np.std(I_tensor, axis=0)
    
    return I_avg, I_std, I_tensor

def _frame_moments(args):
    u_filename, traj_filename, selection, frame_idx = args
    import MDAnalysis as mda
    u = mda.Universe(u_filename, traj_filename)
    ag = u.select_atoms(selection)
    u.trajectory[frame_idx]
    n_molecules = len(ag.residues)
    moments = np.zeros((n_molecules, 3))
    for i, res in enumerate(ag.residues):
        res_pos = res.atoms.positions - res.atoms.center_of_mass()
        res_masses = res.atoms.masses
        I = calculate_molecule_inertia_tensor(res_pos, res_masses)
        principal_moments = np.sort(np.linalg.eigvalsh(I))
        moments[i, :] = principal_moments
    return moments


def calculate_time_averaged_moments(atomgroup, n_workers=1) -> np.ndarray:
    """
    Parallel version: Calculate the time-averaged principal moments of inertia for each molecule in an AtomGroup over the trajectory.
    Returns an array of shape (n_molecules, 3) with the average principal moments for each molecule.
    """
    u = atomgroup.universe
    n_frames = len(u.trajectory)
    n_molecules = len(atomgroup.residues)
    selection = atomgroup.selections[0] if hasattr(atomgroup, 'selections') else 'all'
    if hasattr(atomgroup, 'string'):
        selection = atomgroup.string
    u_filename = u.filename
    traj_filename = u.trajectory.filename
    args_list = [(u_filename, traj_filename, selection, i) for i in range(n_frames)]
    all_moments = np.zeros((n_frames, n_molecules, 3))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for iframe, moments in enumerate(tqdm(executor.map(_frame_moments, args_list), total=n_frames, desc='Calculating principal moments of inertia (parallel)', leave=False)):
            all_moments[iframe, :, :] = moments
    time_averaged = np.mean(all_moments, axis=0)
    return time_averaged


def sample_and_average_principal_moments(atomgroup, fraction=0.05, plot_histogram=False):
    """
    Sample a fraction of frames, calculate per-molecule principal moments for each, and average.
    Optionally plot a histogram of the three principal moments across molecules.
    Returns the averaged (n_molecules, 3) array.
    """
    n_frames = len(atomgroup.universe.trajectory)
    n_molecules = len(atomgroup.residues)
    # Determine number of frames to sample
    n_samples = min(int(np.ceil(fraction * n_frames)), n_frames)
    rng = np.random.default_rng()
    # Randomly select frame indices to sample
    sampled_frames = rng.choice(n_frames, size=n_samples, replace=False)
    # Allocate array to store principal moments for each sample
    I_lk_samples = np.zeros((n_samples, n_molecules, 3))
    # Loop over sampled frames and calculate principal moments
    for i, frame_idx in enumerate(tqdm(sampled_frames, desc="Calculating principal moments of inertia:", leave=False)):
        atomgroup.universe.trajectory[frame_idx]
        I_lk_frame, _, _ = calculate_average_moments(atomgroup)
        I_lk_samples[i, :, :] = I_lk_frame
    # Average over sampled frames
    I_lk = np.mean(I_lk_samples, axis=0)
    # Optionally plot histogram of the three principal moments
    if plot_histogram:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        plt.hist(I_lk[:,0], bins=30, alpha=0.5, label='I₁', color='tab:blue')
        plt.hist(I_lk[:,1], bins=30, alpha=0.5, label='I₂', color='tab:orange')
        plt.hist(I_lk[:,2], bins=30, alpha=0.5, label='I₃', color='tab:green')
        plt.xlabel('Principal moment of inertia (amu·Å²)')
        plt.ylabel('Count')
        plt.title('Distribution of Principal Moments of Inertia (sampled frames)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('I_histogram.png')
        plt.close()
    return I_lk
