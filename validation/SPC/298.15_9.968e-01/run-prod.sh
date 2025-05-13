#!/bin/bash

# Exit on error
set -e

gmx grompp -f prod.mdp -c eq.gro -t eq.cpt -p topol.top -o prod.tpr -maxwarn 1

# Run the simulation
gmx mdrun -s prod.tpr -v