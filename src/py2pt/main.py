"""
Main entry point for the Py2PT package.

This script calculates velocity power spectra and entropy using the Two-Phase Thermodynamics (2PT) method from molecular dynamics trajectories.
It parses command-line arguments, loads the trajectory, performs velocity decomposition, computes density of states, and outputs entropy and spectra results.
"""
#==========================================================================
# Imports
#==========================================================================
# Dependency imports
import yaml
import argparse
import sys
import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis import transformations

# Internal imports
from . import __version__
from .output        import *
from .constants     import *
from .velocity      import *
from .inertia       import *
from .spectrum      import *
from .entropy       import *


def main():
    """
    Main function to run the Py2PT workflow from the command line.

    Parses options, loads the trajectory and topology, 
    performs velocity decomposition, computes density of states, and calculates entropy using the 2PT method. 
    Results are printed and saved to output files.
    """
    #==========================================================================
    # Parse command-line arguments
    #==========================================================================
    parser = argparse.ArgumentParser(
        description="Calculate density of states spectra and entropy of a molecular dynamics trajectory using 2PT method."
    )
    parser.add_argument('-c', '--config', type=str, help='Path to the Py2PT config file (default: Py2PT.yml)', default='Py2PT.yml')
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    # Read the config file
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Extract values from the config file
    temperature = config.get('temperature', 300.0)
    topology_file = config.get('topology', 'topol.tpr')
    trajectory_file = config.get('trajectory', 'traj.trr')
    nproc = config.get('nproc', 1)
    sigma = config.get('sigma', 1)
    FILTERING = config.get('filter', True)
    filter_window = config.get('filter_window', 5)
    CONSTRAINED_BONDS = config.get('constraints', False)
    RENORMALIZE_DOS = config.get('renormalize', True)

    # Check if topology and trajectory files exist
    if not os.path.isfile(topology_file):
        print(f"Error: Topology file '{topology_file}' does not exist.")
        sys.exit(1)
    if not os.path.isfile(trajectory_file):
        print(f"Error: Trajectory file '{trajectory_file}' does not exist.")
        sys.exit(1)


    #==========================================================================
    # Opening print
    #==========================================================================

    print("\n" + "="*80)
    print(f"Launching Py2PT version '{__version__}'")
    print(f"Loading Topology: {topology_file}")
    print(f"Loading Trajectory: {trajectory_file}")
    print(f"nproc: {nproc}")

    #==========================================================================
    # Load and prepare trajectory
    #==========================================================================
    # Load the simulation universe
    try:
        u = mda.Universe(topology_file, trajectory_file)
    except Exception as e:
        print("Could not load Universe.")
        print("Error:", e)
        sys.exit(1)

    # Unwrap the trajectory to remove periodic boundary artifacts
    print("INFO: Unwrapping trajectory in MDAnalysis")
    workflow = [transformations.unwrap(u.atoms)]
    u.trajectory.add_transformations(*workflow)

    # Selection of atoms by string
    ag = u.select_atoms("all")

    #==========================================================================
    # System properties
    #==========================================================================
    # Basic properties
    dt = u.trajectory.dt
    masses = u.atoms.masses
    total_mass = np.sum(masses)  # in amu
    n_atoms = len(ag)
    n_molecules = len(ag.residues)
    atoms_per_molecule = int(n_atoms/n_molecules)
    molecule_mass = total_mass/n_molecules
    volume = mda.lib.mdamath.box_volume(u.dimensions)   # Calculate volume in Å³
    mass_density = (total_mass * amu) / (volume * angstrom**3)
    
    # Print system information
    print("\n" + "="*80)
    print(" "*30 + "TRAJECTORY INFORMATION")
    print("="*80)
    print(f"{'Time step:':<25} {dt:.3f} ps")
    print(f"{'Number of atoms:':<25} {n_atoms}")
    print(f"{'Number of molecules:':<25} {n_molecules}")
    print(f"{'Atoms per molecule:':<25} {atoms_per_molecule}")
    print(f"{'Mass per molecule:':<25} {molecule_mass:.2f} amu")
    print(f"{'Simulation volume:':<25} {volume:.2f} Å³")
    print(f"{'Mass density:':<25} {mass_density/1000:.3e} g/cm³")
    print(f"{'Temperature:':<25} {temperature} K")
    # Print user-defined options
    print("\n" + "="*80)
    print(" "*30 + "USER DEFINED OPTIONS")
    print("="*80)
    print(f"{'FFT filtering:':<25} {FILTERING}")
    print(f"{'FFT filter window:':<25} {filter_window}")
    print(f"{'Constrained bonds:':<25} {CONSTRAINED_BONDS}")
    print(f"{'Renomalize DOS:':<25} {RENORMALIZE_DOS}")

    #==========================================================================
    # Velocity decomposition and power spectra
    #==========================================================================
    # Set is_linear as appropriate for your system (False for most molecules)
    is_linear = False

    # Decompose velocities into translational, rotational, and vibrational components
    print("\n" + "="*80)
    print(" "*15 + "VELOCITY DECOMPOSITION & DENSITY OF STATES SPECTRA")
    print("="*80)
    print("\nDecomposing velocities in trajectory...")
    vt_all, _, vv_all, omega_all, _ = decompose_trajectory_velocities(ag, is_linear=is_linear, n_workers=nproc)
    print("Finished velocity decomposition.")
    # omega_all has shape (n_frames, n_residues, 3) and contains the angular velocity vector for each molecule (residue) per frame

    #==========================================================================
    # Two-Phase Thermodynamics (2PT) Analysis
    #==========================================================================
    print("\n" + "="*80)
    print(" "*25 + "TWO-PHASE THERMODYNAMICS ANALYSIS")
    print("="*80)

    # Principal moments of inertia calculation
    # Use trajectory-averaged moments for entropy calculation
    I_lk = sample_and_average_principal_moments(ag, fraction=0.01, plot_histogram=True)
    I_1, I_2, I_3 = np.mean(I_lk, axis=0)
    print(f"Principal moments of inertia: I_1 = {I_1:.4f}, I_2 = {I_2:.4f}, I_3 = {I_3:.4f} amu·Å²")

    # Compute the density of states spectra for each velocity component
    print(f"Calculating power spectra at T = {temperature} K...")
    if FILTERING:
        print("INFO: Using Blackman windowing before FFT")
        print("INFO: Using Savitsky-Golay filtering after FFT")
    freqs, tDOS = density_of_states(vt_all, masses, dt, temperature, FILTERING, filter_window)
    _, rDOS     = rotational_density_of_states(omega_all, [I_1,I_2,I_3], dt, temperature, FILTERING, filter_window)
    _, vDOS     = density_of_states(vv_all, masses, dt, temperature, FILTERING, filter_window)

    # Integrate the power spectra
    vt_integral = np.trapz(tDOS, freqs)
    vr_integral = np.trapz(rDOS, freqs)
    vv_integral = np.trapz(vDOS, freqs)

    # Print DOS spectra results
    print("\nIntegrated DOS Spectra:")
    print("-"*40)
    print(f"{'Translational:':<20} {vt_integral:>12.4f}")
    print(f"{'Rotational:':<20} {vr_integral:>12.4f}")
    print(f"{'Vibrational:':<20} {vv_integral:>12.4f}")
    print("-"*40)
    print(f"{'Total:':<20} {(vt_integral + vr_integral + vv_integral):>12.4f}")
    
    if RENORMALIZE_DOS:
        # DOS re-normalization
        print(f"\nINFO: Re-normalizing DOS by scaling factor to match theoretical DoS")
        print(f"-- Scale the DOS to set the integral equal to 'real' degrees of freedom")
        print(f"-- 3*nMol (for translation, rotation) and 3*nMol*(nAtomPerMol-2) for vibration")
        print(f"-- This can lead to better convergence of entropy for short MD trajectories")
        print(f"-- Reference: M.A. Caro, T. Laurila, and O. Lopez-Acevedo, The Journal of Chemical Physics 145, (2016).")

        tDOS *= 3*n_molecules/np.trapz(tDOS, freqs)
        rDOS *= 3*n_molecules/np.trapz(rDOS, freqs)
        if CONSTRAINED_BONDS == False:
            vDOS *= 3*n_molecules*(atoms_per_molecule-2)/np.trapz(vDOS, freqs)

    # Calculate the total DOS
    DOS_total = tDOS+rDOS+vDOS

    # Decompose density of states using 2PT method
    print("\nDecomposing density of states...")
    print("-"*40)
    Delta_tr, f_tr, s0_tr, DOS_tr_g, DOS_tr_s = decompose_translational_dos(
        freqs, tDOS, temperature, volume, n_molecules, molecule_mass
    )
    Delta_rot, f_rot, s0_rot, DOS_rot_g, DOS_rot_s = decompose_rotational_dos(
        freqs, rDOS, temperature, volume, n_molecules, molecule_mass
    )
    Delta_vib, f_vib, s0_vib, _, _ = decompose_vibrational_dos(
            freqs, vDOS
    )

    # Calculate entropies (skip zero frequency)
    print("\nCalculating component entropies...")
    S_tr = calculate_translational_entropy(
        freqs[1:], DOS_tr_s[1:], DOS_tr_g[1:],
        f_tr, Delta_tr, molecule_mass, n_molecules, volume, temperature
    )

    S_rot = calculate_rotational_entropy(
        freqs[1:], DOS_rot_s[1:], DOS_rot_g[1:],
        I_1, I_2, I_3, sigma, temperature
    )

    S_vib = calculate_vibrational_entropy(
        freqs[1:], vDOS[1:], temperature
    )
    
    # Save and plot spectra
    spectra_list = [
        {'freqs': freqs, 'DOS': tDOS, 'label': 'Translational'},
        {'freqs': freqs, 'DOS': DOS_tr_g, 'label': 'Translational (g)'},
        {'freqs': freqs, 'DOS': DOS_tr_s, 'label': 'Translational (s)'},
        {'freqs': freqs, 'DOS': rDOS, 'label': 'Rotational'},
        {'freqs': freqs, 'DOS': DOS_rot_g, 'label': 'Rotational (g)'},
        {'freqs': freqs, 'DOS': DOS_rot_s, 'label': 'Rotational (s)'},
        {'freqs': freqs, 'DOS': vDOS, 'label': 'Vibrational'},
        {'freqs': freqs, 'DOS': DOS_total, 'label': 'Total'},
    ]
    save_spectra_txt(spectra_list, filename='density_of_states.txt')
    plot_spectra(spectra_list, filename='density_of_states.png', xlim=[0,1500])

    # Calculate totals for each entropy-related row
    s0_total = s0_tr + s0_rot + s0_vib
    Delta_total = Delta_tr + Delta_rot + Delta_vib
    f_total = f_tr + f_rot + f_vib
    S_tr_mol = S_tr / n_molecules
    S_rot_mol = S_rot / n_molecules
    S_vib_mol = S_vib / n_molecules
    S_total_mol = (S_tr + S_rot + S_vib) / n_molecules

    # Print 2PT results
    print("\n2PT Results:")
    print("-"*81)
    print(f"{'Parameter':<15}{'Translational':>16}{'Rotational':>16}{'Vibrational':>16}{'Total':>16}")
    print("-"*81)
    print(f"{'S_0 (ps)':<15}{s0_tr:>16.2f}{s0_rot:>16.2f}{s0_vib:>16.2f}{s0_total:>16.2f}")
    print(f"{'Delta':<15}{Delta_tr:>16.5f}{Delta_rot:>16.5f}{Delta_vib:>16.5f}{Delta_total:>16.5f}")
    print(f"{'Fluidicity':<15}{f_tr:>16.5f}{f_rot:>16.5f}{f_vib:>16.5f}{f_total:>16.5f}")
    print(f"{'S (J/molK)':<15}{S_tr_mol:>16.2f}{S_rot_mol:>16.2f}{S_vib_mol:>16.2f}{S_total_mol:>16.2f}")
    print("-"*81)
    # Save entropy results with total column as sum of three components per row
    np.savetxt(
        'entropy.txt',
        np.column_stack((
            [S_tr_mol, S_rot_mol, S_vib_mol, S_total_mol],
        )),
        header='(J/mol K)   Translational   Rotational   Vibrational   Total',
        fmt='%.2f'
    )

    # Check for negative entropy (which indicates a problem)
    if S_tr < 0:
        print("\n" + "!"*80)
        print("WARNING: Translational entropy is negative!")
        print("This indicates a problem with the calculation.")
        print("Please report this bug with a reproducible example.")
        print("!"*80)
        sys.exit(4)

if __name__ == "__main__":
    main()
