using BSON: @load
using LinearAlgebra
using Trixi: @muladd

# Load pre-trained machine learning models for source term prediction
base_path = joinpath(dirname(Base.find_package("TrixiMomentEquations")), "Machine_learning_data")

# Declare global variables to store loaded ML models
global model_Ar_1D_s4
@load joinpath(base_path, "model_Ar_1D_s4.bson") model_Ar_1D_s4

global model_Ar_1D_s5
@load joinpath(base_path, "model_Ar_1D_s5.bson") model_Ar_1D_s5

@muladd begin

# Define the G13 equations structure with machine learning source terms in 1D
struct G13EquationsML1DDIM{RealT <: Real} <: AbstractG13Equations{1, 5}
    μref::RealT        # Reference viscosity
    Tref::RealT        # Reference temperature
    dref::RealT        # Reference molecular diameter
    Mref::RealT        # Reference molecular mass
    ω::RealT           # Viscosity exponent

    # Constructor for G13 equations with ML source terms
    function G13EquationsML1DDIM(μref::RealT, Tref::RealT, dref::RealT, Mref::RealT, ω::RealT) where RealT <: Real
        new{RealT}(μref, Tref, dref, Mref, ω)
    end
end

# Variable names for conservative and primitive variables
Trixi.varnames(::typeof(cons2cons), ::G13EquationsML1DDIM) = ("ρ", "ρv1", "ρvv_3p", "ρv1v1_p1", "ρvvv1_3pv1_2p1v1_2q1")
Trixi.varnames(::typeof(cons2prim), ::G13EquationsML1DDIM) = ("ρ", "v1", "p", "p1", "q1")

# Compute the flux vector for the G13 equations
@inline function Trixi.flux(u, orientation::Integer, equations::G13EquationsML1DDIM)
    ρ, ρv1, ρvv_3p, ρv1v1_p1, ρvvv1_3pv1_2p1v1_2q1 = u
    ρ, v1, p, p1, q1 = Trixi.cons2prim(u, equations)

    θ = p / ρ  # Temperature in energy units
    v = v1     # Velocity component

    # Compute flux components for each conservation equation
    f1 = ρ * v1                                   # Mass flux
    f2 = ρ * v1 * v1 + p1                         # Momentum flux
    f3 = ρ * v * v * v1 + 3.0 * p * v1 + 2.0 * p1 * v1 + 2.0 * q1  # Energy flux
    f4 = ρ * v1 * v1 * v1 + 3.0 * p1 * v1 + 1.2 * q1              # Third moment flux
    f5 = (ρ * v * v + 3.0 * p + 4.0 * p1) * v1 * v1 + v * v * p1 + 6.4 * q1 * v1 + 7.0 * θ * p1 - 2.0 * θ * p  # Fourth moment flux
    
    return SVector(f1, f2, f3, f4, f5)
end

# Normalize variables for ML model input
@inline function normalize_hyperbolic(u, equations::G13EquationsML1DDIM)
    ρ, v1, p, p1, q1 = Trixi.cons2prim(u, equations)
    
    # Calculate reference velocity using ideal gas law
    vref = sqrt(equations.Tref * 8314.5 / equations.Mref)

    # Calculate temperature in Kelvin
    T = p / ρ * equations.Mref / 8314.5
    
    # Normalize variables
    v1_N = v1 / vref                    # Normalized velocity
    T_N = T / equations.Tref            # Normalized temperature
    p_N = p / (ρ * vref^2)              # Normalized pressure
    σ1_N = (p1 - p) / (ρ * vref^2)      # Normalized stress
    q1_N = q1 / (ρ * vref^3)            # Normalized heat flux
    
    return SVector(T_N, p_N, σ1_N, q1_N)
end

# Predict source terms using machine learning models
@inline function predict_source_terms(all, equations::G13EquationsML1DDIM)
    T_N, p_N, σ1_N, q1_N = all

    # Prepare input data for ML models
    input_data = reshape([T_N, p_N, σ1_N, q1_N], 4, 1)

    # Get predictions from ML models
    predicted_outputs_s4 = model_Ar_1D_s4(Float32.(input_data))
    predicted_outputs_s5 = model_Ar_1D_s5(Float32.(input_data))
    
    s4_mod_N = predicted_outputs_s4[1]  # Normalized source term for equation 4
    s5_mod_N = predicted_outputs_s5[1]  # Normalized source term for equation 5

    return SVector(s4_mod_N, s5_mod_N)
end

# Compute relaxation source terms with ML predictions
@inline function relaxation_source(u, x, t, equations::G13EquationsML1DDIM)
    ρ, v1, p, p1, q1 = Trixi.cons2prim(u, equations)
    
    v = v1
    # Calculate reference velocity
    vref = sqrt(equations.Tref * 8314.5 / equations.Mref)
    
    # Normalize variables for ML input
    normalized_vars = normalize_hyperbolic(u, equations)
    T_N, p_N, σ1_N, q1_N = normalized_vars

    # Get ML predictions for source terms
    s4_mod_N, s5_mod_N = predict_source_terms(normalized_vars, equations)

    θ = p/ρ  # Temperature in energy units
    T = θ * equations.Mref / 8314.5  # Temperature in Kelvin
    μ = equations.μref * (T / equations.Tref)^equations.ω  # Dynamic viscosity

    # Calculate moments for source term scaling
    g4 = ρ * v1 * v1 * v1 + 3.0 * p1 * v1 + 1.2 * q1
    g5 = (ρ * v * v + 3.0 * p + 4.0 * p1) * v1 * v1 + v * v * p1 + 6.4 * q1 * v1 + 7.0 * θ * p1 - 2.0 * θ * p

    # Calculate mean free path
    mfp = (equations.Mref * 1.66053906660e-27) / (sqrt(2) * pi * equations.dref^2 * T_N^(-equations.ω + 0.5) * ρ)

    # Scale ML predictions to physical units
    s4 = ρ * vref^2 * p / μ * s4_mod_N * 1.0
    s5_ori = ρ * vref^3 * p / μ * s5_mod_N * 10.0 # the original range is so big to be applied directly

    # Apply physical constraints to source terms to mitigate unstable predictions.
    # This is necessary because the initial condition, which has significant physical discontinuities, 
    # is far from the training data distribution, leading to highly inaccurate results.

    if (p1 - p) > 0
        s4 = clamp(s4, -p/μ * (p1 - p) * 1.5, -p/μ * (p1 - p) * 0.9)
    else 
        s4 = clamp(s4, -p/μ * (p1 - p) * 0.9, -p/μ * (p1 - p) * 1.5)
    end

    if 2/3 * q1 > 0
        s5_ori = clamp(s5_ori, -p/μ * 2 * 2/3 * q1 * 1.5, -p/μ * 2 * 2/3 * q1 * 0.9)
    else 
        s5_ori = clamp(s5_ori, -p/μ * 2 * 2/3 * q1 * 0.9, -p/μ * 2 * 2/3 * q1 * 1.5)
    end

    # Final source term for equation 5
    s5 = s5_ori + 2.0 * v1 * s4
     
    return SVector(0, 0, 0, s4, s5)  # Only last two equations have source terms
end

# Convert conservative variables to primitive variables
@inline function Trixi.cons2prim(u, equations::G13EquationsML1DDIM)
    ρ, ρv1, ρvv_3p, ρv1v1_p1, ρvvv1_3pv1_2p1v1_2q1 = u
    v1 = ρv1 / ρ                    # Velocity
    v = v1
    p = (ρvv_3p - ρ * v * v) / 3.0  # Pressure
    p1 = ρv1v1_p1 - ρ * v1 * v1     # Normal stress
    q1 = 0.5 * (ρvvv1_3pv1_2p1v1_2q1 - ρ * v * v * v1 - 3.0 * p1 * v1 - 2.0 * p1 * v1)  # Heat flux
    
    return SVector(ρ, v1, p, p1, q1)
end

# Convert primitive variables to conservative variables
@inline function Trixi.prim2cons(prim, equations::G13EquationsML1DDIM)
    ρ, v1, p, p1, q1 = prim
    v = v1
    ρv1 = ρ * v1
    ρvv_3p = ρ * v * v + 3.0 * p
    ρv1v1_p1 = ρ * v1 * v1 + p1
    ρvvv1_3pv1_2p1v1_2q1 = ρ * v * v * v1 + 3.0 * p * v1 + 2.0 * p1 * v1 + 2.0 * q1
    
    return SVector(ρ, ρv1, ρvv_3p, ρv1v1_p1, ρvvv1_3pv1_2p1v1_2q1)
end

# Calculate maximum absolute wave speeds
@inline function Trixi.max_abs_speeds(u, equations::G13EquationsML1DDIM)
    ρ, ρv1, ρvv_3p, ρvv_p1, ρvvv_3pv_2p1v_2σv2_2q1 = u
    v1 = ρv1 / ρ
    v = v1
    p = (ρvv_3p - ρ * v * v) / 3.0
    
    c = sqrt(5 / 3 * p / ρ)  # Sound speed
    return abs(v1) + c       # Maximum wave speed
end

# Calculate maximum absolute speed for Riemann solver (naive approach)
@inline function Trixi.max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::G13EquationsML1DDIM)
    ρ_ll, v1_ll, p_ll, p1_ll, q1_ll  = Trixi.cons2prim(u_ll, equations)
    ρ_rr, v1_rr, p_rr, p1_rr, q1_rr  = Trixi.cons2prim(u_rr, equations)

    # Calculate sound speeds on both sides
    c_ll = sqrt(5 / 3 * p_ll / ρ_ll)
    c_rr = sqrt(5 / 3 * p_rr / ρ_rr)

    λ_max = max(abs(v1_ll), abs(v1_rr)) + max(c_ll, c_rr)
    return λ_max
end

# Calculate minimum and maximum wave speeds for Riemann solver
@inline function Trixi.min_max_speed_naive(u_ll, u_rr, orientation::Integer, equations::G13EquationsML1DDIM)
    ρ_ll, v1_ll, p_ll, p1_ll, q1_ll  = Trixi.cons2prim(u_ll, equations)
    ρ_rr, v1_rr, p_rr, p1_rr, q1_rr  = Trixi.cons2prim(u_rr, equations)

    λ_min = v1_ll - sqrt(5 / 3 * p_ll / ρ_ll)  # Minimum wave speed
    λ_max = v1_rr + sqrt(5 / 3 * p_rr / ρ_rr)  # Maximum wave speed

    return λ_min, λ_max
end

# Calculate pressure from conservative variables
@inline function Trixi.pressure(u, equations::G13EquationsML1DDIM)
    ρ, ρv1, ρvv_3p, ρv1v1_p1, ρvvv1_3pv1_2p1v1_2q1 = u

    v1 = ρv1 / ρ
    v = v1
    p = (ρvv_3p - ρ * v * v) / 3.0
    return p
end

# Calculate temperature in energy units
@inline function Trixi.temperature(u, equations::G13EquationsML1DDIM)
    θ = pressure(u, equations) / u[1]  # θ = p/ρ
    return θ
end

# Extract density from conservative variables
@inline function Trixi.density(u, equations::G13EquationsML1DDIM)
    ρ = u[1]
    return ρ
end

# Calculate density times pressure
@inline function Trixi.density_pressure(u, equations::G13EquationsML1DDIM)
    return Trixi.density(u, equations) * Trixi.pressure(u, equations)
end

# Outflow boundary condition implementation
function boundary_condition_outflow(u_inner, orientation, normal_direction, x, t,
  surface_flux_function, equations::G13EquationsML1DDIM)
    # Calculate boundary flux using internal solution state only
    flux = Trixi.flux(u_inner, normal_direction, equations)

    return flux
end

# Convert conservative variables to entropy variables (placeholder implementation)
@inline function Trixi.cons2entropy(u, equations::G13EquationsML1DDIM)
    w1=w2=w3=w4=w5=0.0  # Placeholder values
    return SVector(w1, w2, w3, w4, w5)
end

end # @muladd