#!/bin/bash

# Exit on error
set -e

# Define file names
TPR="npt.tpr"
GRO="solvate.gro"
TOP="topol.top"
MDP="npt.mdp"

# Remove existing TPR file if it exists
if [ -f $TPR ]; then
    echo "Removing existing $TPR file..."
    rm $TPR
fi

# Create the binary input file (.tpr) for GROMACS
echo "Preparing simulation input file..."
gmx grompp -f $MDP -c $GRO -p $TOP -o $TPR -maxwarn 1

# Run the simulation
echo "Starting NPT simulation..."
gmx mdrun -deffnm npt -v

echo "NPT simulation complete."


