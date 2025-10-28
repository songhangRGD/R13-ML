using OrdinaryDiffEq
using Trixi
using TrixiMomentEquations
using JuMP, NLopt, SpecialFunctions
using Plots

###############################################################################
# semidiscretization of the compressible Euler equations
μref = 2.117e-5
Tref = 273.15
dref = 4.11e-10
Mref = 39.948
ω = 0.81

equations = G13EquationsML1DDIM(μref, Tref, dref, Mref, ω)
equations_parabolic = R13EquationsML1DDIM(equations, gradient_variables=GradientVariablesPrimitive())

function initial_condition_shock_wave(x, t, equations::G13EquationsML1DDIM)

  gamma = 5.0/3.0
  Ma_1 = 9.0
  Ma_2 = sqrt((1.0+(gamma-1.0)/2.0*Ma_1*Ma_1)/(gamma*Ma_1*Ma_1-(gamma-1.0)/2.0))
  p_ratio = (1.0+gamma*Ma_1*Ma_1)/(1.0+gamma*Ma_2*Ma_2)
  ρ_ratio = ((gamma-1.0)+(gamma+1.0)*p_ratio)/((gamma+1.0)+(gamma-1.0)*p_ratio)
  
  θref = Tref*8314.5/equations.Mref
  ρ_1 = 1.3324517321071778e21*equations.Mref*1.66053906660e-27
  p_1 = ρ_1*θref
  p1_1 = ρ_1*θref
  q1_1 = 0.0
  v1_1 = Ma_1*sqrt(gamma*p_1/ρ_1)

  p_2 = p_ratio*p_1
  ρ_2 = ρ_ratio*ρ_1
  v1_2 = Ma_2*sqrt(gamma*p_2/ρ_2)
  p1_2 = p_ratio*p_1
  q1_2 = 0.0

  if (x[1] <= 0.00)
      ρ      = ρ_1
      v1     = v1_1
      p      = p_1
      p1     = p1_1
      q1     = q1_1
  else
      ρ      = ρ_2
      v1     = v1_2
      p      = p_2
      p1     = p1_2
      q1     = q1_2
  end
  
  return prim2cons(SVector(ρ, v1, p, p1, q1),equations)
end



initial_condition = initial_condition_shock_wave

coordinates_min = (-0.05,)
coordinates_max = ( 0.05,)

boundary_conditions_hyperbolic = ( x_pos = boundary_condition_outflow,
                                   x_neg = boundary_condition_outflow)

boundary_conditions_parabolic = (  x_pos = BoundaryConditionOutflowParabolic,
                                   x_neg = BoundaryConditionOutflowParabolic)

surface_flux =  flux_lax_friedrichs
volume_flux  =  flux_lax_friedrichs                              

basis = LobattoLegendreBasis(3)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)


mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=7,
                periodicity=(false,false),
                n_cells_max=100000)


semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, solver, 
boundary_conditions=(boundary_conditions_hyperbolic, boundary_conditions_parabolic), source_terms=relaxation_source)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.0001)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(alive_interval=100)

stepsize_callback = StepsizeCallback(cfl=0.02)

save_solution = SaveSolutionCallback(dt=tspan[2]/1,
                                     save_initial_solution = false,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

callbacks = CallbackSet(alive_callback, stepsize_callback, save_solution)

###############################################################################
# run the simulation


sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            maxiters = 99999999,
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

include("trixi2tec.jl") # provide the output files which can be read by Tecplot

# create the output directory if it does not already exist
if !isdir("out/")
    mkpath("out/")
end

idx = length(sol.t)
filename = joinpath("out/", "R13_ML_Ma9.tec")
trixi2tec(sol.u[idx], semi, filename, solution_variables=cons2prim)
plot(sol)
