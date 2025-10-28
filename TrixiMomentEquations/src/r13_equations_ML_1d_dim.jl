using BSON: @load
using LinearAlgebra
using Statistics
using Trixi: @muladd

# Load pre-trained machine learning models for higher-order moment prediction
base_path = joinpath(dirname(Base.find_package("TrixiMomentEquations")), "Machine_learning_data")

# Declare global variables to store loaded ML models for parabolic terms
global model_Ar_1D_mv1v1v1
@load joinpath(base_path, "model_Ar_1D_mv1v1v1.bson") model_Ar_1D_mv1v1v1

global model_Ar_1D_mvvv1v1
@load joinpath(base_path, "model_Ar_1D_mvvv1v1.bson") model_Ar_1D_mvvv1v1

@muladd begin

# Define R13 equations structure with machine learning for parabolic terms in 1D
struct R13EquationsML1DDIM{E <: G13EquationsML1DDIM, GradientVariables} <: Trixi.AbstractEquationsParabolic{1, 5, GradientVariables}
    equations_hyperbolic::E           # Hyperbolic part of the equations (G13)
    gradient_variables::GradientVariables # Variables used for gradient computation
end

# Constructor for R13 equations with ML parabolic terms
function R13EquationsML1DDIM(equations::G13EquationsML1DDIM; gradient_variables = GradientVariablesPrimitive())
    R13EquationsML1DDIM{typeof(equations), typeof(gradient_variables)}(equations, gradient_variables)
end

# Variable names mapping (delegate to hyperbolic equations)
varnames(var_map, eqs::R13EquationsML1DDIM) = varnames(var_map, eqs.equations_hyperbolic)

# Define gradient variable transformation for parabolic terms
function Trixi.gradient_variable_transformation(::R13EquationsML1DDIM)
    cons2prim  # Use primitive variables for gradient computation
end

# Convert derivative to primitive variables (identity transformation in this case)
@inline function convert_derivative_to_primitive(u, gradient, ::R13EquationsML1DDIM)
    return gradient
end

# Convert transformed variables to primitive variables (identity transformation)
@inline function convert_transformed_to_primitive(u_transformed, ::R13EquationsML1DDIM)
    return u_transformed
end

# Normalize variables and gradients for ML model input in parabolic equations
@inline function normalize_parabolic(u, gradients, orientation::Integer, equations::R13EquationsML1DDIM)
    # Extract primitive variables and their gradients
    ρ, v1, p, p1, q1 = convert_transformed_to_primitive(u, equations)
    dρdx, dv1dx, dpdx, dp1dx, dq1dx = convert_derivative_to_primitive(u, gradients, equations)

    # Calculate reference velocity using ideal gas law
    vref = sqrt(equations.equations_hyperbolic.Tref * 8314.5 / equations.equations_hyperbolic.Mref)

    # Calculate temperature and its gradient
    T = p / ρ * equations.equations_hyperbolic.Mref / 8314.5
    dTdx = (ρ * dpdx - p * dρdx) / (ρ * ρ) * equations.equations_hyperbolic.Mref / 8314.5

    # Normalize state variables
    T_N = T / equations.equations_hyperbolic.Tref
    p_N = p / (ρ * vref^2)
    σ1_N = (p1 - p) / (ρ * vref^2)
    q1_N = q1 / (ρ * vref^3)

    # Calculate mean free path for scaling
    mfp = (equations.equations_hyperbolic.Mref * 1.66053906660e-27) / 
          (sqrt(2) * pi * equations.equations_hyperbolic.dref^2 * T_N^(-equations.equations_hyperbolic.ω + 0.5) * ρ)

    # Normalize gradients using mean free path scaling. The dvdx should be taken into consideration, but the one-dimensional shock samples 
    # cannot have positive values, which causes incorrect prediction when the test case requires positive dv/dx. Hence, it is not contained
    dρdx_N = dρdx / (ρ / mfp)
    dTdx_N = dTdx / (equations.equations_hyperbolic.Tref / mfp)
    dpdx_N = dpdx / (ρ * vref^2 / mfp)
    dσdx_N = (dp1dx - dpdx) / (ρ * vref^2 / mfp)
    dq1dx_N = dq1dx / (ρ * vref^3 / mfp)

    return SVector(T_N, p_N, σ1_N, q1_N, dρdx_N, dTdx_N, dpdx_N, dσdx_N, dq1dx_N)
end

# Predict higher-order moments using machine learning models
@inline function predict_high_order_moments(all, equations::R13EquationsML1DDIM)
    T_N, p_N, σ1_N, q1_N, dρdx_N, dTdx_N, dpdx_N, dσdx_N, dq1dx_N = all

    # Prepare input data for ML models (9 features)
    input_data = reshape([T_N, p_N, σ1_N, q1_N, dρdx_N, dTdx_N, dpdx_N, dσdx_N, dq1dx_N], 9, 1)

    # Get predictions from ML models for higher-order moments
    predicted_outputs_mv1v1v1 = model_Ar_1D_mv1v1v1(Float32.(input_data))
    predicted_outputs_mvvv1v1 = model_Ar_1D_mvvv1v1(Float32.(input_data))

    mv1v1v1_mod_N = predicted_outputs_mv1v1v1[1]  # Normalized third-order moment
    mvvv1v1_mod_N = predicted_outputs_mvvv1v1[1]  # Normalized fourth-order moment

    return SVector(mv1v1v1_mod_N, mvvv1v1_mod_N)
end

# Compute parabolic flux terms for R13 equations with ML predictions
function Trixi.flux(u, gradients, orientation::Integer, equations::R13EquationsML1DDIM)
    ρ, v1, p, p1, q1 = convert_transformed_to_primitive(u, equations)
    dρdx, dv1dx, dpdx, dp1dx, dq1dx = convert_derivative_to_primitive(u, gradients, equations)

    θ = p / ρ  # Temperature in energy units
    T = θ * equations.equations_hyperbolic.Mref / 8314.5  # Temperature in Kelvin
    μ = equations.equations_hyperbolic.μref * (T / equations.equations_hyperbolic.Tref)^equations.equations_hyperbolic.ω  # Viscosity
    𝜈 = p / μ  # Kinematic viscosity-like term

    v = v1
    # Calculate reference velocities
    vref = sqrt(equations.equations_hyperbolic.Tref * 8314.5 / equations.equations_hyperbolic.Mref)
    vref_local = sqrt(T * 8314.5 / equations.equations_hyperbolic.Mref)  # Local reference velocity

    # Normalize variables for ML input
    normalized_vars = normalize_parabolic(u, gradients, orientation, equations)
    T_N, p_N, σ1_N, q1_N, dρdx_N, dTdx_N, dpdx_N, dσdx_N, dq1dx_N = normalized_vars

    # Calculate mean free path
    mfp = (equations.equations_hyperbolic.Mref * 1.66053906660e-27) / 
          (sqrt(2) * pi * equations.equations_hyperbolic.dref^2 * T_N^(-equations.equations_hyperbolic.ω + 0.5) * ρ)

    # Get ML predictions for higher-order moments
    mv1v1v1_mod_N, mvvv1v1_mod_N = predict_high_order_moments(normalized_vars, equations)

    # Scale ML predictions to physical units
    m111 = μ * vref^2 / mfp * mv1v1v1_mod_N * 1.0   # Third-order moment m111
    R11  = μ * vref^3 / mfp * mvvv1v1_mod_N * 10.0  # Fourth-order moment R11, the original range is so big to be applied directly

    # Apply physical constraints to high-order moments to mitigate unstable predictions.
    # This is necessary because the initial condition, which has significant physical discontinuities, 
    # is far from the training data distribution, leading to highly inaccurate results.

    limit_m111 = min(20.0 * μ/ρ * (3.0/5.0 * abs(dp1dx - dpdx)), 
                    max(6.0 * μ/ρ * (3.0/5.0 * abs(dp1dx - dpdx)), 3.0 * μ * vref_local^2 / mfp))
    
    limit_R11 = min(40.0 * μ/ρ * 9.0/5.0 * abs(dq1dx), 
                   max(15.0 * μ/ρ * 9.0/5.0 * abs(dq1dx), 150.0 * μ * vref_local^3 / mfp))

    # Apply constraints to m111 based on gradient conditions
    if abs(2.0 * μ/ρ * 3.0/5.0 * (dp1dx - dpdx)) < 0.03 * μ * vref_local^2 / mfp && 
       abs(2.0 * μ/ρ * 3.0/5.0 * dpdx) > 0.05 * μ * vref_local^2 / mfp
        m111 = clamp(m111, -0.08 * μ * vref_local^2 / mfp, 0.08 * μ * vref_local^2 / mfp)
    elseif 2.0 * μ/ρ * 3.0/5.0 * (dp1dx - dpdx) > 0
        m111 = clamp(m111, -limit_m111, 0.0)
    else
        m111 = clamp(m111, 0.0, limit_m111)
    end

    # Apply constraints to R11 based on gradient conditions
    if abs(4.0 * μ/ρ * 9.0/5.0 * dq1dx) < 3.0 * μ * vref_local^3 / mfp && 
       abs(2.0 * μ/ρ * 3.0/5.0 * dpdx) > 0.05 * μ * vref_local^2 / mfp
        R11 = clamp(R11, -8.0 * μ * vref_local^3 / mfp, 8.0 * μ * vref_local^3 / mfp)
    elseif 4.0 * μ/ρ * 9.0/5.0 * dq1dx > 0
        R11 = clamp(R11, -limit_R11, 0.0)
    else
        R11 = clamp(R11, 0.0, limit_R11)
    end 
   
    # Compute parabolic flux components (only non-zero for higher moments)
    f1 = 0.0  # Mass flux (no parabolic contribution)
    f2 = 0.0  # Momentum flux (no parabolic contribution)
    f3 = 0.0  # Energy flux (no parabolic contribution)
    f4 = -m111  # Flux for third-order moment equation
    f5 = -2.0 * m111 * v1 - R11  # Flux for fourth-order moment equation

    return SVector(f1, f2, f3, f4, f5)
end

# Outflow boundary condition implementation for parabolic terms (gradient operator)
@inline function BoundaryConditionOutflowParabolic(flux_inner, u_inner, orientation::Integer, direction, x, t,
    operator_type::Trixi.Gradient, equations_parabolic::R13EquationsML1DDIM)
    return flux_inner  # Use internal flux for outflow
end

# Outflow boundary condition implementation for parabolic terms (divergence operator)
@inline function BoundaryConditionOutflowParabolic(flux_inner, nothing, orientation::Integer, direction, x, t,
    operator_type::Trixi.Divergence, equations_parabolic::R13EquationsML1DDIM)
    return flux_inner  # Use internal flux for outflow
end

# Constant Dirichlet boundary condition implementation (gradient operator)
@inline function (boundary_condition::BoundaryConditionConstantDirichlet)(flux_inner, u_inner, orientation::Integer,
    direction, x, t, operator_type::Trixi.Gradient, equations_parabolic::R13EquationsML1DDIM)
    value = boundary_condition.boundary_value
    bc = SVector(value, value, value, value, value)  # Constant value for all variables
    return bc
end

# Constant Dirichlet boundary condition implementation (divergence operator)
@inline function (boundary_condition::BoundaryConditionConstantDirichlet)(flux_inner, nothing, orientation::Integer,
    direction, x, t, operator_type::Trixi.Divergence, equations_parabolic::R13EquationsML1DDIM)
    value = boundary_condition.boundary_value
    bc = SVector(value, value, value, value, value)  # Constant value for all variables
    return bc
end

# Convert conservative variables to primitive variables
@inline function Trixi.cons2prim(u, equations::R13EquationsML1DDIM)
    ρ, ρv1, ρvv_3p, ρv1v1_p1, ρvvv1_3pv1_2p1v1_2q1 = u
    v1 = ρv1 / ρ                    # Velocity
    v = v1
    p = (ρvv_3p - ρ * v * v) / 3.0  # Pressure
    p1 = ρv1v1_p1 - ρ * v1 * v1     # Normal stress
    q1 = 0.5 * (ρvvv1_3pv1_2p1v1_2q1 - ρ * v * v * v1 - 3.0 * p1 * v1 - 2.0 * p1 * v1)  # Heat flux
    
    return SVector(ρ, v1, p, p1, q1)
end

# Delegate thermodynamic calculations to hyperbolic equations
@inline Trixi.temperature(u, equations::R13EquationsML1DDIM) = Trixi.temperature(u, equations.equations_hyperbolic)
@inline Trixi.density(u, equations::R13EquationsML1DDIM) = Trixi.density(u, equations.equations_hyperbolic)
@inline Trixi.pressure(u, equations::R13EquationsML1DDIM) = Trixi.pressure(u, equations.equations_hyperbolic)
@inline Trixi.prim2cons(prim, equations::R13EquationsML1DDIM) = Trixi.prim2cons(prim, equations.equations_hyperbolic)

end # @muladd