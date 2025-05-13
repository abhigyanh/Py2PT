#!/bin/bash

# Exit on error
set -e

# Define file names
NPT_GRO="npt.gro"
NPT_CPT="npt.cpt"
TOP="topol.top"
MDP="nvt.mdp"
NVT_TPR="nvt.tpr"

# Remove existing TPR file if it exists
if [ -f $NVT_TPR ]; then
    echo "Removing existing $NVT_TPR file..."
    rm $NVT_TPR
fi

# Create the binary input file (.tpr) for GROMACS
echo "Preparing NVT simulation input file..."
gmx grompp -f $MDP -c $NPT_GRO -t $NPT_CPT -p $TOP -o $NVT_TPR -maxwarn 1

# Run the simulation
echo "Starting NVT simulation..."
gmx mdrun -s $NVT_TPR -v

echo "NVT simulation complete."
