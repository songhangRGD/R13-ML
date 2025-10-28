# Reproducibility Repository for the paper "Extraction of Moment Closures for Strongly Non-Equilibrium Flows via Machine Learning"

# Overview
Welcome to the reproducibility repository for the paper "Reconstruction of Moment Equations for Strongly Non-Equilibrium Flows via Machine Learning" by Hang Song, Satyvir Singh, Manuel Torrilhon and Semih Cayci. This repository contains the necessary files to reproduce the results of shock at Ma=9 presented in the paper.

# Package Support
The julia version applied in this research is 1.10.1, while the version of Trixi.jl is v0.12.6. Other packages required should be installed through "julia".
```julia
using Pkg
Pkg.add(["OrdinaryDiffEq", "CUDA", "DiffEqBase", "SciMLBase", "Flux", "JuMP", "NLopt", "SpecialFunctions", "LinearAlgebra", "Statistics", "DiffEqCallbacks", "BSON", "Plots"])
```

# Training Data from DSMC
The Training_Ar folder contains datasets for machine learning. All files named _input.dat are normalized data, while tmp_flowvss. files are the original data generated from DSMC. After training, the resulting models are used within the TrixiMomentEquations framework and can be found in "TrixiMomentEquations\src\Machine_learning_data".

# Install Trixi.jl
Please follow the link https://trixi-framework.github.io/TrixiDocumentation/stable/tutorials/first_steps/getting_started/, install julia and Trixi.jl framework based on the tutorial.

# Install TrixiMomentEquations
Start julia with "julia".
```julia
using Pkg
Pkg.add(url="https://github.com/songhangRGD/R13-ML.git", subdir="TrixiMomentEquations")
```
# Introduction of TrixiMomentEquations
TrixiMomentEquations is a julia package with moment equations for the simulation of rarefied gas dynamics. Only the R13-ML model is contained in this package, the prediction of high-order moments is in "r13_equations_ML_1d_dim.jl", while the collision integrals are in "g13_equations_ML_1d_dim.jl". Four machine learning models are loaded from the folder "TrixiMomentEquations\src\Machine_learning_data" and applied for the simulation.

# Run the Simulation
Download and enter the folder "example", start julia with 16 threads to enable parallel computing with "julia --threads 16".
```julia --threads 16
using Trixi, TrixiMomentEquations
trixi_include("elixir_r13_ML_shock_Ar_dim.jl")
```
The file "out/R13_ML_Ma9.tec" is the simulation result of R13-ML visualized by Tecplot. You can compare it with the DSMC result "DSMC.dat". The common output format of Trixi.jl is ".h5", with a mesh file "mesh.h5"