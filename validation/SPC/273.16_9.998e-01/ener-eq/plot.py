import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Set style
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = (15, 10)

# Get all xvg files
xvg_files = glob.glob('*.xvg')

# Create subplots
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

# Plot each file
for idx, file in enumerate(sorted(xvg_files)):
    # Read data
    data = np.loadtxt(file, comments=['#', '@'])
    time = data[:, 0]
    value = data[:, 1]
    
    # Get quantity name from filename
    quantity = os.path.basename(file).replace('.xvg', '')
    
    # Plot
    axes[idx].plot(time, value, 'b-', linewidth=1)
    axes[idx].set_title(quantity.capitalize())
    axes[idx].set_xlabel('Time (ps)')
    axes[idx].grid(True)
    
    # Add units based on quantity
    if quantity == 'temperature':
        axes[idx].set_ylabel('Temperature (K)')
    elif quantity == 'pressure':
        axes[idx].set_ylabel('Pressure (bar)')
    elif quantity == 'density':
        axes[idx].set_ylabel('Density (kg/m³)')
    else:
        axes[idx].set_ylabel('Energy (kJ/mol)')

# Adjust layout
plt.tight_layout()
plt.savefig('energy_plots.png', dpi=300, bbox_inches='tight')
plt.close() 