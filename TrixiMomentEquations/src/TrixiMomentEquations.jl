module TrixiMomentEquations # Extends Trixi with R13-ML models
# installation via julia REPL in outer directory: pkg> dev ./TrixiMomentEquations

export relaxation_source
export BoundaryConditionOutflowParabolic, BoundaryConditionConstantDirichlet, boundary_condition_outflow
export boundary_condition_driven_wall, boundary_condition_fixed_wall
export cons2output
export normalize_hyperbolic, normalize_parabolic
export G13EquationsML1DDIM, R13EquationsML1DDIM

using Flux
using BSON
using Trixi
using Trixi: @muladd
@muladd begin
#

#= Please note that no effort was spent on optimization.
Quite the opposite: legibility is preferred. =#

struct BoundaryConditionConstantDirichlet{RealT <: Real}
    boundary_value::RealT
end

struct GradientVariablesPrimitive end

abstract type AbstractG13Equations{NDIMS, NVARS} <: Trixi.AbstractEquations{NDIMS, NVARS} end

# Machine learning
include("g13_equations_ML_1d_dim.jl")
include("r13_equations_ML_1d_dim.jl")

end #@muladd

end # module TrixiMomentEquations
