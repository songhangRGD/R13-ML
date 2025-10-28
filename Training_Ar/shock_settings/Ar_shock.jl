mfp=0.001
omega=0.81
d_ref=4.11e-10
T_ref=273.15
T_1=273.15
Ma_1=10.0
γ=5.0/3.0
m=39.948*1.66053906660e-27

n_1=1/(sqrt(2.0)*pi*d_ref^2*(T_ref/T_1)^(omega-0.5)*mfp)
Ma_2=sqrt((1.0+(γ-1.0)/2.0*Ma_1*Ma_1)/(γ*Ma_1*Ma_1-(γ-1.0)/2.0))
p_ratio=1.0+2.0*γ/(γ+1.0)*(Ma_1*Ma_1-1.0)
rho_ratio=((γ+1.0)*Ma_1*Ma_1)/(2.0+(γ-1.0)*Ma_1*Ma_1)
T_ratio=p_ratio/rho_ratio

n_2=n_1*rho_ratio

rho_1=n_1*m
rho_2=n_2*m

p_1=n_1*1.380649e-23*T_1
p_2=p_ratio*p_1

T_2=T_1*T_ratio

v_1=Ma_1*sqrt(γ*p_1/rho_1)
v_2=v_1/rho_ratio

println("Ma_1 = $Ma_1")
println("n_1 = $n_1")
println("n_2 = $n_2")
println("rho_1 = $rho_1")
println("rho_2 = $rho_2")
println("p_1 = $p_1")
println("p_2 = $p_2")
println("T_1 = $T_1")
println("T_2 = $T_2")
println("v_1 = $v_1")
println("v_2 = $v_2")
