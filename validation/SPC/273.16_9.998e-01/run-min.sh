#!/bin/bash

# Exit on error
set -e

# Define file names
TPR="min.tpr"
GRO="solvate.gro"
TOP="topol.top"
MDP="min.mdp"

# Remove existing TPR file if it exists
if [ -f $TPR ]; then
    echo "Removing existing $TPR file..."
    rm $TPR
fi

# Create the binary input file (.tpr) for GROMACS
echo "Preparing minimization input file..."
gmx grompp -f $MDP -c $GRO -p $TOP -o $TPR -maxwarn 1

# Run the minimization
echo "Running energy minimization..."
gmx mdrun -deffnm min -v

echo "Energy minimization completed. Output files: min.gro, min.edr, min.log" 