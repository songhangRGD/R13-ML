using BSON: @load
using LinearAlgebra
using Trixi: @muladd

# 加载模型
base_path = joinpath(dirname(Base.find_package("TrixiMomentEquations")), "Machine_learning_data_Ar_1d")

# 声明全局变量，用于存储加载的模型
global model_Ar_1D_s4
@load joinpath(base_path, "model_Ar_1D_s4.bson") model_Ar_1D_s4

global model_Ar_1D_s5
@load joinpath(base_path, "model_Ar_1D_s5.bson") model_Ar_1D_s5

@muladd begin

struct G13EquationsML1DDIM{RealT <: Real} <: AbstractG13Equations{1, 5}
    μref::RealT        
    Tref::RealT
    dref::RealT
    Mref::RealT        
    ω::RealT           

    function G13EquationsML1DDIM(μref::RealT, Tref::RealT, dref::RealT, Mref::RealT, ω::RealT) where RealT <: Real
        new{RealT}(μref, Tref, dref, Mref, ω)
    end
end

Trixi.varnames(::typeof(cons2cons), ::G13EquationsML1DDIM) = ("ρ", "ρv1", "ρvv_3p", "ρv1v1_p1", "ρvvv1_3pv1_2p1v1_2q1")
Trixi.varnames(::typeof(cons2prim), ::G13EquationsML1DDIM) = ("ρ", "v1", "p", "p1", "q1")

@inline function Trixi.flux(u, orientation::Integer, equations::G13EquationsML1DDIM)
    ρ, ρv1, ρvv_3p, ρv1v1_p1, ρvvv1_3pv1_2p1v1_2q1 = u
    ρ, v1, p, p1, q1 = Trixi.cons2prim(u, equations)

    θ = p / ρ
    v = v1

    f1 = ρ * v1 
    f2 = ρ * v1 * v1 + p1 
    f3 = ρ * v * v * v1 + 3.0 * p * v1 + 2.0 * p1 * v1 + 2.0 * q1 
    f4 = ρ * v1 * v1 * v1 + 3.0 * p1 * v1 + 1.2 * q1
    f5 = (ρ * v * v + 3.0 * p + 4.0 * p1) * v1 * v1 + v * v * p1 + 6.4 * q1 * v1 + 7.0 * θ * p1 - 2.0 * θ * p 
    return SVector(f1, f2, f3, f4, f5)
end

@inline function normalize_hyperbolic(u, equations::G13EquationsML1DDIM)
    ρ, v1, p, p1, q1 = Trixi.cons2prim(u, equations)
    
    vref = sqrt(equations.Tref * 8314.5 / equations.Mref)

    T = p / ρ * equations.Mref / 8314.5
    
    v1_N = v1 / vref
    T_N = T / equations.Tref
    p_N = p / (ρ * vref^2)
    σ1_N = (p1 - p) / (ρ * vref^2)
    q1_N = q1 / (ρ * vref^3)
    
    return SVector(T_N, p_N, σ1_N, q1_N)
end

@inline function predict_source_terms(all, equations::G13EquationsML1DDIM)
    T_N, p_N, σ1_N, q1_N = all

    input_data = reshape([T_N, p_N, σ1_N, q1_N], 4, 1)

    predicted_outputs_s4 = model_Ar_1D_s4(Float32.(input_data))
    predicted_outputs_s5 = model_Ar_1D_s5(Float32.(input_data))
    
    s4_mod_N = predicted_outputs_s4[1]
    s5_mod_N = predicted_outputs_s5[1]

    return SVector(s4_mod_N, s5_mod_N)
end

@inline function _collision_source_from_prediction(u, s4_mod_N, s5_mod_N,
                                                   equations::G13EquationsML1DDIM)
    ρ, v1, p, p1, q1 = Trixi.cons2prim(u, equations)

    v = v1
    vref = sqrt(equations.Tref * 8314.5 / equations.Mref)

    θ = p/ρ
    T = θ*equations.Mref/8314.5
    μ = equations.μref*(T/equations.Tref)^equations.ω

    g4 = ρ * v1 * v1 * v1 + 3.0 * p1 * v1 + 1.2 * q1
    g5 = (ρ * v * v + 3.0 * p + 4.0 * p1) * v1 * v1 + v * v * p1 + 6.4 * q1 * v1 + 7.0 * θ * p1 - 2.0 * θ * p

    s4 = ρ * vref^2 * p/μ * s4_mod_N * 1.0
    s5_ori = ρ * vref^3 * p/μ * s5_mod_N * 10.0

    if (p1 - p)>0
        s4 = clamp(s4, -p/μ*(p1 - p)*1.5, -p/μ*(p1 - p)*0.9)
    else 
        s4 = clamp(s4, -p/μ*(p1 - p)*0.9, -p/μ*(p1 - p)*1.5)
    end

    if 2/3*q1>0
        s5_ori = clamp(s5_ori, -p/μ*2*2/3*q1*1.5, -p/μ*2*2/3*q1*0.9)
    else 
        s5_ori = clamp(s5_ori, -p/μ*2*2/3*q1*0.9, -p/μ*2*2/3*q1*1.5)
    end

    s5 = s5_ori + 2.0 * v1 * s4
     
    return SVector(0, 0, 0, s4, s5)
end

@inline function relaxation_source(u, x, t, equations::G13EquationsML1DDIM)
    normalized_vars = normalize_hyperbolic(u, equations)
    s4_mod_N, s5_mod_N = predict_source_terms(normalized_vars, equations)

    return _collision_source_from_prediction(u, s4_mod_N, s5_mod_N, equations)
end

# Trixi's default source kernel calls `relaxation_source` once per DG node. This
# specialization assembles every local DG degree of freedom as one batch, invokes
# each neural network once, and then applies the original pointwise constraints.
function Trixi.calc_sources!(du, u, t,
                             source_terms::typeof(relaxation_source),
                             equations::G13EquationsML1DDIM,
                             dg::Trixi.DG, cache)
    n_nodes = size(u, 2)
    n_elements = size(u, 3)
    n_points = n_nodes * n_elements
    n_points == 0 && return nothing

    input_data = Matrix{Float32}(undef, 4, n_points)

    @inbounds for element in 1:n_elements, i in 1:n_nodes
        column = i + (element - 1) * n_nodes
        u_local = Trixi.get_node_vars(u, equations, dg, i, element)
        T_N, p_N, σ1_N, q1_N = normalize_hyperbolic(u_local, equations)

        input_data[1, column] = T_N
        input_data[2, column] = p_N
        input_data[3, column] = σ1_N
        input_data[4, column] = q1_N
    end

    predicted_outputs_s4 = model_Ar_1D_s4(input_data)
    predicted_outputs_s5 = model_Ar_1D_s5(input_data)

    length(predicted_outputs_s4) == n_points ||
        throw(DimensionMismatch("s4 model returned $(length(predicted_outputs_s4)) values for $n_points DG nodes"))
    length(predicted_outputs_s5) == n_points ||
        throw(DimensionMismatch("s5 model returned $(length(predicted_outputs_s5)) values for $n_points DG nodes"))

    @inbounds for element in 1:n_elements, i in 1:n_nodes
        column = i + (element - 1) * n_nodes
        u_local = Trixi.get_node_vars(u, equations, dg, i, element)
        du_local = _collision_source_from_prediction(u_local,
                                                     predicted_outputs_s4[column],
                                                     predicted_outputs_s5[column],
                                                     equations)
        Trixi.add_to_node_vars!(du, du_local, equations, dg, i, element)
    end

    return nothing
end

@inline function Trixi.cons2prim(u, equations::G13EquationsML1DDIM)
    ρ, ρv1, ρvv_3p, ρv1v1_p1, ρvvv1_3pv1_2p1v1_2q1 = u
    v1 = ρv1 / ρ
    v = v1
    p = (ρvv_3p - ρ * v * v) / 3.0
    p1 = ρv1v1_p1 - ρ * v1 * v1
    q1 = 0.5 * (ρvvv1_3pv1_2p1v1_2q1 - ρ * v * v * v1 - 3.0 * p1 * v1 - 2.0 * p1 * v1)
    return SVector(ρ, v1, p, p1, q1)
end


@inline function Trixi.prim2cons(prim, equations::G13EquationsML1DDIM)
    ρ, v1, p, p1, q1 = prim
    v = v1
    ρv1 = ρ * v1
    ρvv_3p = ρ * v * v + 3.0 * p
    ρv1v1_p1 = ρ * v1 * v1 + p1
    ρvvv1_3pv1_2p1v1_2q1 = ρ * v * v * v1 + 3.0 * p * v1 + 2.0 * p1 * v1 + 2.0 * q1
    return SVector(ρ, ρv1, ρvv_3p, ρv1v1_p1, ρvvv1_3pv1_2p1v1_2q1)
end

@inline function Trixi.max_abs_speeds(u, equations::G13EquationsML1DDIM)
    ρ, ρv1, ρvv_3p, ρvv_p1, ρvvv_3pv_2p1v_2σv2_2q1 = u
    v1 = ρv1 / ρ
    v = v1
    p = (ρvv_3p - ρ * v * v) / 3.0
    
    c = sqrt(5 / 3 * p / ρ)
    return abs(v1) + c
end

@inline function Trixi.max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::G13EquationsML1DDIM)
    ρ_ll, v1_ll, p_ll, p1_ll, q1_ll  = Trixi.cons2prim(u_ll, equations)
    ρ_rr, v1_rr, p_rr, p1_rr, q1_rr  = Trixi.cons2prim(u_rr, equations)

    # Calculate sound speeds
    c_ll = sqrt(5 / 3 * p_ll / ρ_ll)
    c_rr = sqrt(5 / 3 * p_rr / ρ_rr)

    λ_max = max(abs(v1_ll), abs(v1_rr)) + max(c_ll, c_rr)
    return λ_max
end

@inline function Trixi.min_max_speed_naive(u_ll, u_rr, orientation::Integer, equations::G13EquationsML1DDIM)
    ρ_ll, v1_ll, p_ll, p1_ll, q1_ll  = Trixi.cons2prim(u_ll, equations)
    ρ_rr, v1_rr, p_rr, p1_rr, q1_rr  = Trixi.cons2prim(u_rr, equations)

    λ_min = v1_ll - sqrt(5 / 3 * p_ll / ρ_ll)
    λ_max = v1_rr + sqrt(5 / 3 * p_rr / ρ_rr)

    return λ_min, λ_max
end

@inline function Trixi.pressure(u, equations::G13EquationsML1DDIM)
    ρ, ρv1, ρvv_3p, ρv1v1_p1, ρvvv1_3pv1_2p1v1_2q1 = u

    v1 = ρv1 / ρ
    v = v1
    p = (ρvv_3p - ρ * v * v) / 3.0
    return p
end

# Calculate the temperature θ in energy units
@inline function Trixi.temperature(u, equations::G13EquationsML1DDIM)
    θ = pressure(u, equations) / u[1] # p = ρθ
    return θ
end

@inline function Trixi.density(u, equations::G13EquationsML1DDIM)
    ρ = u[1]
    return ρ
end

@inline function Trixi.density_pressure(u, equations::G13EquationsML1DDIM)
    return Trixi.density(u, equations) * Trixi.pressure(u, equations)
end

function boundary_condition_outflow(u_inner, orientation, normal_direction, x, t,
  surface_flux_function, equations::G13EquationsML1DDIM)
    # Calculate the boundary flux entirely from the internal solution state
    flux = Trixi.flux(u_inner, normal_direction, equations)

    return flux
end

@inline function Trixi.cons2entropy(u, equations::G13EquationsML1DDIM)
    w1=w2=w3=w4=w5=0.0
    return SVector(w1, w2, w3, w4, w5)
end

end # @muladd
