using BSON: @load
using LinearAlgebra
using Statistics

using Trixi: @muladd

# 加载模型
base_path = joinpath(dirname(Base.find_package("TrixiMomentEquations")), "Machine_learning_data_Ar_1d")

# 声明全局变量，用于存储加载的模型
global model_Ar_1D_mv1v1v1
@load joinpath(base_path, "model_Ar_1D_mv1v1v1.bson") model_Ar_1D_mv1v1v1

global model_Ar_1D_mvvv1v1
@load joinpath(base_path, "model_Ar_1D_mvvv1v1.bson") model_Ar_1D_mvvv1v1

@muladd begin

# 定义 R13 方程的结构体
struct R13EquationsML1DDIM{E <: G13EquationsML1DDIM, GradientVariables} <: Trixi.AbstractEquationsParabolic{1, 5, GradientVariables}
    equations_hyperbolic::E    
    gradient_variables::GradientVariables 
end

# 构造函数
function R13EquationsML1DDIM(equations::G13EquationsML1DDIM; gradient_variables = GradientVariablesPrimitive())
    R13EquationsML1DDIM{typeof(equations), typeof(gradient_variables)}(equations, gradient_variables)
end

# 变量名称映射
varnames(var_map, eqs::R13EquationsML1DDIM) = varnames(var_map, eqs.equations_hyperbolic)

# 梯度变量的转换
function Trixi.gradient_variable_transformation(::R13EquationsML1DDIM)
    cons2prim
end

@inline function convert_derivative_to_primitive(u, gradient, ::R13EquationsML1DDIM)
    return gradient
end

@inline function convert_transformed_to_primitive(u_transformed, ::R13EquationsML1DDIM)
    return u_transformed
end

# Trixi 0.13 passes one gradient vector per spatial direction as a tuple. Keep
# accepting a bare vector as well for direct calls and older Trixi releases.
@inline _gradient_1d(gradients::Tuple) = only(gradients)
@inline _gradient_1d(gradients) = gradients

# 对抛物方程变量进行归一化
@inline function normalize_parabolic(u, gradients, orientation::Integer, equations::R13EquationsML1DDIM)
    ρ, v1, p, p1, q1 = convert_transformed_to_primitive(u, equations)
    gradients_1d = _gradient_1d(gradients)
    dρdx, dv1dx, dpdx, dp1dx, dq1dx = convert_derivative_to_primitive(u, gradients_1d,
                                                                       equations)

    vref = sqrt(equations.equations_hyperbolic.Tref * 8314.5 / equations.equations_hyperbolic.Mref)

    T = p / ρ * equations.equations_hyperbolic.Mref / 8314.5
    dTdx = (ρ * dpdx - p * dρdx) / (ρ * ρ) * equations.equations_hyperbolic.Mref / 8314.5

    T_N = T / equations.equations_hyperbolic.Tref
    p_N = p / (ρ * vref^2)
    σ1_N = (p1 - p) / (ρ * vref^2)
    q1_N = q1 / (ρ * vref^3)

    mfp = (equations.equations_hyperbolic.Mref*1.66053906660e-27)/(sqrt(2)*pi*equations.equations_hyperbolic.dref^2*T_N^(-equations.equations_hyperbolic.ω+0.5)*ρ)

    dρdx_N = dρdx / (ρ / mfp)
    dTdx_N = dTdx / (equations.equations_hyperbolic.Tref / mfp)
    dpdx_N = dpdx / (ρ * vref^2 / mfp)
    dσdx_N = (dp1dx - dpdx) / (ρ * vref^2 / mfp)
    dq1dx_N = dq1dx / (ρ * vref^3 / mfp)

    return SVector(T_N, p_N, σ1_N, q1_N, dρdx_N, dTdx_N, dpdx_N, dσdx_N, dq1dx_N)
end

# 高阶矩预测函数
@inline function predict_high_order_moments(all, equations::R13EquationsML1DDIM)

    T_N, p_N, σ1_N, q1_N, dρdx_N, dTdx_N, dpdx_N, dσdx_N, dq1dx_N = all

    input_data = reshape([T_N, p_N, σ1_N, q1_N, dρdx_N, dTdx_N, dpdx_N, dσdx_N, dq1dx_N], 9, 1)

    predicted_outputs_mv1v1v1 = model_Ar_1D_mv1v1v1(Float32.(input_data))
    predicted_outputs_mvvv1v1 = model_Ar_1D_mvvv1v1(Float32.(input_data))

    mv1v1v1_mod_N = predicted_outputs_mv1v1v1[1]
    mvvv1v1_mod_N = predicted_outputs_mvvv1v1[1]

    return SVector(mv1v1v1_mod_N, mvvv1v1_mod_N)
end

# Convert normalized neural-network outputs to the physical viscous flux. Scalar
# and batched inference both use this function, preserving the same constraints.
@inline function _parabolic_flux_from_prediction(u, gradients,
                                                 mv1v1v1_mod_N, mvvv1v1_mod_N,
                                                 equations::R13EquationsML1DDIM)
    ρ, v1, p, p1, q1 = convert_transformed_to_primitive(u, equations)
    gradients_1d = _gradient_1d(gradients)
    dρdx, dv1dx, dpdx, dp1dx, dq1dx = convert_derivative_to_primitive(u, gradients_1d,
                                                                       equations)

    θ = p / ρ
    T = θ * equations.equations_hyperbolic.Mref / 8314.5
    μ = equations.equations_hyperbolic.μref * (T / equations.equations_hyperbolic.Tref) ^ equations.equations_hyperbolic.ω
    𝜈 = p / μ

    v = v1
    vref = sqrt(equations.equations_hyperbolic.Tref * 8314.5 / equations.equations_hyperbolic.Mref)
    vref_local = sqrt(T * 8314.5 / equations.equations_hyperbolic.Mref)

    T_N = T / equations.equations_hyperbolic.Tref

    mfp = (equations.equations_hyperbolic.Mref*1.66053906660e-27)/(sqrt(2)*pi*equations.equations_hyperbolic.dref^2*T_N^(-equations.equations_hyperbolic.ω+0.5)*ρ)

    m111 = μ * vref^2 / mfp * mv1v1v1_mod_N * 1.0
    R11  = μ * vref^3 / mfp * mvvv1v1_mod_N * 10.0

    limit_m111 = min(20.0 * μ/ρ * (3.0/5.0 * abs(dp1dx-dpdx)), max(7.0 * μ/ρ * (3.0/5.0 * abs(dp1dx-dpdx)), 2.5 * μ * vref_local^2 / mfp))
    limit_R11 = min(40.0* μ/ρ * 9.0/5.0 * abs(dq1dx), max(15.0 * μ/ρ * 9.0/5.0 * abs(dq1dx), 150.0 * μ * vref_local^3 / mfp))

    if abs(2.0*μ/ρ*3.0/5.0*(dp1dx-dpdx)) < 0.02 * ρ * vref_local^2 / mfp && abs(2.0*μ/ρ*3.0/5.0*dpdx) > 0.05 * ρ * vref_local^2 / mfp
        m111 = clamp(m111, -0.05 * ρ * vref_local^2 / mfp, 0.05 * ρ * vref_local^2 / mfp)
    elseif 2.0*μ/ρ*3.0/5.0*(dp1dx-dpdx) > 0
        m111 = clamp(m111, -limit_m111, 0.0)
    else
        m111 = clamp(m111, 0.0, limit_m111)
    end

    if abs(4.0*μ/ρ*9.0/5.0*dq1dx) < 2.0 * μ * vref_local^3 / mfp && abs(2.0*μ/ρ*3.0/5.0*dpdx) > 0.05 * ρ * vref_local^2 / mfp
        R11 = clamp(R11, -5.0 * μ * vref_local^3 / mfp, 5.0 * μ * vref_local^3 / mfp)
    elseif 4.0*μ/ρ*9.0/5.0*dq1dx > 0
        R11 = clamp(R11, -limit_R11, 0.0)
    else
        R11 = clamp(R11, 0.0, limit_R11)
    end 

    f1 = 0.0
    f2 = 0.0
    f3 = 0.0
    f4 = -m111
    f5 = -2.0 * m111 * v1 - R11

    return SVector(f1, f2, f3, f4, f5)
end

# Pointwise fallback used outside Trixi's DG volume kernel.
function Trixi.flux(u, gradients, orientation::Integer, equations::R13EquationsML1DDIM)
    normalized_vars = normalize_parabolic(u, gradients, orientation, equations)
    mv1v1v1_mod_N, mvvv1v1_mod_N = predict_high_order_moments(normalized_vars, equations)

    return _parabolic_flux_from_prediction(u, gradients,
                                           mv1v1v1_mod_N, mvvv1v1_mod_N,
                                           equations)
end


# Assemble the complete one-dimensional DG field into a 9 x N feature matrix,
# invoke each moment model once, then apply the original scaling and constraints
# independently at every DG node.
function Trixi.calc_viscous_fluxes!(flux_viscous, gradients, u_transformed,
                                    mesh::Trixi.TreeMesh{1},
                                    equations::R13EquationsML1DDIM,
                                    dg::Trixi.DG, cache)
    n_nodes = size(u_transformed, 2)
    n_elements = size(u_transformed, 3)
    n_points = n_nodes * n_elements
    n_points == 0 && return nothing

    input_data = Matrix{Float32}(undef, 9, n_points)

    @inbounds for element in 1:n_elements, i in 1:n_nodes
        column = i + (element - 1) * n_nodes
        u_node = Trixi.get_node_vars(u_transformed, equations, dg, i, element)
        gradients_node = Trixi.get_node_vars(gradients, equations, dg, i, element)
        normalized_vars = normalize_parabolic(u_node, gradients_node, 1, equations)

        for feature in 1:9
            input_data[feature, column] = normalized_vars[feature]
        end
    end

    predicted_outputs_mv1v1v1 = model_Ar_1D_mv1v1v1(input_data)
    predicted_outputs_mvvv1v1 = model_Ar_1D_mvvv1v1(input_data)

    length(predicted_outputs_mv1v1v1) == n_points ||
        throw(DimensionMismatch("mv1v1v1 model returned $(length(predicted_outputs_mv1v1v1)) values for $n_points DG nodes"))
    length(predicted_outputs_mvvv1v1) == n_points ||
        throw(DimensionMismatch("mvvv1v1 model returned $(length(predicted_outputs_mvvv1v1)) values for $n_points DG nodes"))

    @inbounds for element in 1:n_elements, i in 1:n_nodes
        column = i + (element - 1) * n_nodes
        u_node = Trixi.get_node_vars(u_transformed, equations, dg, i, element)
        gradients_node = Trixi.get_node_vars(gradients, equations, dg, i, element)
        flux_node = _parabolic_flux_from_prediction(u_node, gradients_node,
                                                    predicted_outputs_mv1v1v1[column],
                                                    predicted_outputs_mvvv1v1[column],
                                                    equations)
        Trixi.set_node_vars!(flux_viscous, flux_node, equations, dg, i, element)
    end

    return nothing
end

# 边界条件实现
@inline function BoundaryConditionOutflowParabolic(flux_inner, u_inner, orientation::Integer, direction, x, t,
    operator_type::Trixi.Gradient, equations_parabolic::R13EquationsML1DDIM)
    return flux_inner
end

@inline function BoundaryConditionOutflowParabolic(flux_inner, nothing, orientation::Integer, direction, x, t,
    operator_type::Trixi.Divergence, equations_parabolic::R13EquationsML1DDIM)
    return flux_inner
end

# 常值 Dirichlet 边界条件
@inline function (boundary_condition::BoundaryConditionConstantDirichlet)(flux_inner, u_inner, orientation::Integer,
    direction, x, t, operator_type::Trixi.Gradient, equations_parabolic::R13EquationsML1DDIM)
    value = boundary_condition.boundary_value
    bc = SVector(value, value, value, value, value)
    return bc
end

@inline function (boundary_condition::BoundaryConditionConstantDirichlet)(flux_inner, nothing, orientation::Integer,
    direction, x, t, operator_type::Trixi.Divergence, equations_parabolic::R13EquationsML1DDIM)
    value = boundary_condition.boundary_value
    bc = SVector(value, value, value, value, value)
    return bc
end

# 变量转换实现
@inline function Trixi.cons2prim(u, equations::R13EquationsML1DDIM)
    ρ, ρv1, ρvv_3p, ρv1v1_p1, ρvvv1_3pv1_2p1v1_2q1 = u
    v1 = ρv1 / ρ
    v = v1
    p = (ρvv_3p - ρ * v * v) / 3.0
    p1 = ρv1v1_p1 - ρ * v1 * v1
    q1 = 0.5 * (ρvvv1_3pv1_2p1v1_2q1 - ρ * v * v * v1 - 3.0 * p1 * v1 - 2.0 * p1 * v1)
    
    return SVector(ρ, v1, p, p1, q1)
end

# 额外的辅助函数
@inline Trixi.temperature(u, equations::R13EquationsML1DDIM) = Trixi.temperature(u, equations.equations_hyperbolic)
@inline Trixi.density(u, equations::R13EquationsML1DDIM) = Trixi.density(u, equations.equations_hyperbolic)
@inline Trixi.pressure(u, equations::R13EquationsML1DDIM) = Trixi.pressure(u, equations.equations_hyperbolic)
@inline Trixi.prim2cons(prim, equations::R13EquationsML1DDIM) = Trixi.prim2cons(prim, equations.equations_hyperbolic)

end # @muladd
