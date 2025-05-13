#!/bin/bash

rm solvate.gro
a=$(python calculate-box-size.py $density)
echo $a
gmx solvate -cs spc216.gro -box $a $a $a -maxsol 500 -o solvate.gro -scale 0.1
