#!/bin/bash

# Exit on error
set -e

# Define file names
TPR="eq.tpr"
TOP="topol.top"
MDP="eq.mdp"
GRO="min.gro"

# Remove existing TPR file if it exists
if [ -f $TPR ]; then
    echo "Removing existing $TPR file..."
    rm $TPR
fi

# Create the binary input file (.tpr) for GROMACS
echo "Preparing simulation input file..."
gmx grompp -f $MDP -c $GRO -p $TOP -o $TPR -maxwarn 1

# Run the simulation
echo "Running equilibration..."
gmx mdrun -deffnm eq -v

echo "Equilibration completed. Output files: eq.gro, eq.edr, eq.log"


