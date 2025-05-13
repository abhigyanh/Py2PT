#!/usr/bin/env python3

import numpy as np
import sys

def calculate_box_size(num_molecules, density_g_cc):
    """
    Calculate the box size (in nm) for a cubic box containing a given number of water molecules
    with a specified density.
    
    Parameters:
    -----------
    num_molecules : int
        Number of water molecules in the box
    density_g_cc : float
        Density in g/cm³
    
    Returns:
    --------
    float
        Box size in nanometers
    """
    # Constants
    water_molar_mass = 15.999 + 1.001 + 1.001  # g/mol
    avogadro_number = 6.02214076e23  # molecules/mol
    
    # Calculate mass of water molecules
    mass_mol = num_molecules / avogadro_number  # mol
    mass_g = mass_mol * water_molar_mass  # g
    
    # Calculate volume in cm³
    volume_cm3 = mass_g / density_g_cc
    
    # Convert to nm³ (1 cm³ = 10^21 nm³)
    volume_nm3 = volume_cm3 * 1e21
    
    # Calculate box size (cube root of volume)
    box_size_nm = np.cbrt(volume_nm3)
    
    return box_size_nm

def main():
    # Check if density is provided as command-line argument
    if len(sys.argv) < 2:
        print("Usage: python calculate-box-size.py <density_g_cm3>")
        sys.exit(1)
    
    try:
        density = float(sys.argv[1])
    except ValueError:
        print("Error: Density must be a number")
        sys.exit(1)
    
    # Parameters
    num_water_molecules = 2000
    
    # Calculate box size
    box_size = calculate_box_size(num_water_molecules, density)
    
    # Print only the box size as a pure float
    print(f"{box_size:.5f}")

if __name__ == "__main__":
    main()
