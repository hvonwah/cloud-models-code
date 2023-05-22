'''
    Rising thermal bubble in an undersaturated atmosphere leading to
    rain, taken from [1]. The microphysics parametrisation is taken from
    the COSMO model [2].

    Note: Due to implementational issues, the x and z axis are swapped.

    Literature
    ----------
    [1] W. W. Grabowski and T. L. Clark. ‘Cloud-Environment Interface
        Instability: Part II: Extension to Three Spatial Dimensions’.
        In: J. Atmos. Sci. 50.4 (1993), pp. 555–573.
    [2] G. Doms et al. COSMO-Model Version 6.00: A Description of the
        Nonhydrostatic Regional COSMO-Model - Part II: Physical Parame-
        trizations. Tech. rep. COSMO Consortium for Small-Scale
        Modelling, (2021).
'''
# ------------------------------ LOAD LIBRARIES -------------------------------
from netgen.occ import Box, OCCGeometry, Z, Y
from netgen.meshing import IdentificationType
from ngsolve import *
from ngsolve.meshes import Make1DMesh, MakeStructured3DMesh
from ngsolve.comp import IntegrationRuleSpace
from ngsolve.fem import NewtonCF
from ngsolve.solvers import Newton
import time as wall_time
from math import isnan, gamma
import os
import pickle
import argparse

SetNumThreads(6)
SetHeapSize(10000000)
ngsglobals.symbolic_integrator_uses_diff = True

time_start = wall_time.time()


# ------------------------- COMMAND-LINE PARAMETERS ---------------------------
parser = argparse.ArgumentParser(description='Rising temerature bubble in 3d with rain',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-hmax', '--mesh_size', type=float, default=200, help='Mesh size')
parser.add_argument('-hm', '--hexes', type=int, default=0, help='Hexahedral mesh mesh if a or tetrahedral mesh if 0')
parser.add_argument('-sm', '--struc', type=int, default=0, help='Structured mesh if 1 or unstructured if 0')
parser.add_argument('-o', '--order', type=int, default=2, help='Order of finite element space')
parser.add_argument('-dt', '--time_step', type=float, default=0.02, help='Time step')
parser.add_argument('-ad', '--diffusion', default=1, type=int, help='Add Péclet scaled diffusion if 1')
parser.add_argument('-dp', '--diffusion_par', default=0.05, type=float, help='Artificial diffusion parameter')
options = vars(parser.parse_args())
print(options)


# -------------------------------- PARAMETERS ---------------------------------
h_max = options['mesh_size']
order = options['order']
hex_mesh = bool(options['hexes'])
struc_mesh = bool(options['struc'])

t_end = 360
dt = options['time_step']

wait_compile = False
compile_flag = True

newton_tol = 1e-9
newton_maxit = 10

artificial_diffusion = bool(options['diffusion'])
scale_diff = options['diffusion_par']
if artificial_diffusion is False:
    scale_diff = 0.0
alpha = 10

pickle_flag = False
pickle_dir = 'pickle_output'
pickle_name = f'rising_thermal_3d_rain_dg_ssprk43_pert_artdiff{scale_diff}'
pickle_name += f'quads{int(hex_mesh)}struc{int(struc_mesh)}'
pickle_name += f'h{h_max}k{order}dt{dt}'
pickle_freq = int(60 / dt)

vtk_flag = False
vtk_dir = 'vtk_output'
vtk_name = pickle_name
vtk_subdiv = 0
vtk_order = max(1, min(2, order))
vtk_freq = int(30 / dt)

restart_flag = False
restart_step = 0

chech_nan_freq = int(0.5 / dt)

# ----------------------------------- DATA ------------------------------------
c_l = 4186                  # Specific heat of water
c_pd = 1004                 # Specific heat of dry air at constant pressure
c_pv = 1885                 # Specific heat of vapour at constant pressure
c_vd = 717                  # Specific heat of dry air at constant volume
c_vv = 1424                 # Specific heat of vapour at constant volume
R_d = c_pd - c_vd           # Gas constant for dry air
R_v = c_pv - c_vv           # Gas constant for water vapour
eps = R_d / R_v
e_ref = 6.107e2             # Saturation vapour pressure with respect to water
L_ref = 2.5e6               # Latent heat of vaporization at T_ref
T_ref = 273.15              # Reference temperature
p_ref = 1e5                 # Reference pressure
N_0 = 8e6                   # Rain water distribution parameter
v_0 = 130                   # Mean rain velocity parameter

g = 9.81                    # Acceleration due to gravity

domain_h = 2400             # Domain height
domain_w = 3600             # Domain width
domain_d = 3600             # Domain depth

T_surf = 283                # Surface temperature
p_surf = 8.5e4              # Surface pressure
stratif = 1.3e-5            # Stratification
humidity_rel0 = 0.2         # Hydrostatic relative humidity
humidity_rel_bar = 0.2      # Background relative humidity field
r1, r2 = 300, 200           # Radii of humidity bubble
x_c, y_c, z_c = domain_w / 2, domain_d / 2, 800  # Center of humidity bubble
Theta0 = T_surf * (p_ref / p_surf)**(R_d / c_pd)  # Dry potential temp. at surf
theta_d0 = Theta0 * exp(stratif * x)  # Hydrostatic dry potential temperature

# Initial humidity function
r = sqrt((z - x_c)**2 + (y - y_c)**2 + (x - z_c)**2)
H_mid = humidity_rel_bar
H_mid += (1 - humidity_rel_bar) * cos(pi * (r - r2) / (2 * (r1 - r2)))**2

H_init = IfPos(r - r1, humidity_rel_bar, IfPos(r - r2, H_mid, 1))


# ------------------------------ MESH & FE-SPACE ------------------------------
if struc_mesh:
    mesh = MakeStructured3DMesh(
        hexes=hex_mesh, periodic_y=True, periodic_z=True,
        nx=int(ceil(domain_h / h_max)), ny=int(ceil(domain_w / h_max)),
        nz=int(ceil(domain_d / h_max)),
        mapping=lambda x, y, z: (domain_h * x, domain_w * y, domain_d * z))
else:
    if hex_mesh:
        str_out = 'WARNING: Unstructured hexahedral meshes not supported!'
        raise NotImplementedError(str_out)
    a = Box((0, 0, 0), (domain_h, domain_w, domain_d))
    a.faces.Min(Y).Identify(a.faces.Max(Y), "periodic_y",
                            IdentificationType.PERIODIC)
    a.faces.Min(Z).Identify(a.faces.Max(Z), "periodic_z",
                            IdentificationType.PERIODIC)
    geo = OCCGeometry(a)
    mesh = Mesh(geo.GenerateMesh(maxh=h_max))

X0 = L2(mesh, order=0)
X1 = L2(mesh, order=order)**7
X2 = L2(mesh, order=order)**3
X = L2(mesh, order=order)**16
IRS = IntegrationRuleSpace(mesh, order=order)**3

print(f'X1 has {X1.ndof} unknowns')

gf_X = GridFunction(X)
gf_rho_d_per, gf_rho_m_per, gf_rho_r_per, gf_rhoU1, gf_rhoU2, gf_rhoU3, \
    gf_E_per, gf_rho_v, gf_rho_c, gf_T, gf_rho_d_0, gf_rho_m_0, gf_rho_r_0, \
    gf_rhoe_0, gf_rho_v_0, gf_T_0 = gf_X.components
gf_rho_d = gf_rho_d_per + gf_rho_d_0
gf_rho_m = gf_rho_m_per + gf_rho_m_0
gf_rho_r = gf_rho_r_per + gf_rho_r_0
gf_rho = gf_rho_d + gf_rho_m + gf_rho_r
gf_rhovel = CF(gf_X.components[3:6])
gf_vel = gf_rhovel / gf_rho
gf_E = gf_E_per + gf_rhoe_0
gf_rhoe = gf_E - (gf_rhoU1**2 + gf_rhoU2**2 + gf_rhoU3**2) / (2 * gf_rho)

gf_X2, gfu_hydro = GridFunction(X2), GridFunction(X2)
gf_IRS = GridFunction(IRS)

mesh1d = Make1DMesh(n=int(ceil(domain_h / h_max)),
                    mapping=lambda x: x * domain_h)

X1_1d = H1(mesh1d, order=order + 1, dirichlet='left')
X2_1d = L2(mesh1d, order=order)**3
X_1d = X1_1d * X2_1d

gfu_1d, gfu_1d_2 = GridFunction(X_1d), GridFunction(X_1d)


# ----------------------------- (BI)LINEAR FORMS ------------------------------
U, V = X.TnT()

n = specialcf.normal(mesh.dim)


def Max(a, b):
    return IfPos(a - b, a, b)


def Min(a, b):
    return IfPos(a - b, b, a)


def Abs(u):
    return IfPos(u, u, -u)


def Pos(u):
    return IfPos(u, u, 0)


# Saturation vapour pressure
def clausius_clapeyron(T):
    p_vs = e_ref * (T / T_ref)**((c_pv - c_l) / R_v)
    p_vs *= exp(((L_ref - (c_pv - c_l) * T_ref) / R_v) * (1 / T_ref - 1 / T))
    return p_vs


# Terminal falling speed of rain
def V_r_func(rho_r, rho_w):
    __v_r = v_0 * gamma(4.5) / 6 * (rho_r / (pi * rho_w * N_0))**(1 / 8)
    return IfPos(rho_r, __v_r, 0)


# Flux function
def F(U):
    rho_d_per, rho_m_per, rho_r_per, rhoU1, rhoU2, rhoU3, E_per, _rho_v,\
        _rho_c, T, _rho_d_0, _rho_m_0, _rho_r_0, E_0, _rho_v_0, T_0 = U

    rho_d_0, rho_r_0, rho_m_0 = Pos(_rho_d_0), Pos(_rho_r_0), Pos(_rho_m_0)
    rho_d = Pos(rho_d_per + rho_d_0)
    rho_m = Pos(rho_m_per + rho_m_0)
    rho_r = Pos(rho_r_per + rho_r_0)
    rho_v = Pos(_rho_v)
    rho_v_0 = Pos(_rho_v_0)
    rho_w = rho_m + rho_r
    rho = rho_d + rho_m + rho_r

    rhovel = U[3:6]
    vel = rhovel / rho
    v_r_vec = CoefficientFunction((V_r_func(rho_r, rho_w), 0, 0))

    p = R_d * rho_d * T + R_v * rho_v * T
    p0 = R_d * rho_d_0 * T_0 + R_v * rho_v_0 * T_0
    p_pert = p - p0

    F1 = rho_d * vel
    F2 = rho_m * vel
    F3 = rho_r * (vel - v_r_vec)
    F4 = OuterProduct(rhovel, vel) - OuterProduct(rho_r * vel, v_r_vec)
    F4 += p_pert * Id(mesh.dim)
    F5 = (E_per + E_0 + p) * vel
    F5 -= (c_l * (T - T_ref) + InnerProduct(vel, vel) / 2) * rho_r * v_r_vec
    Fz = CF((0, 0, 0))

    return CF((F1, F2, F3, F4, F5, Fz, Fz, Fz, Fz, Fz, Fz, Fz, Fz, Fz),
              dims=(16, 3))


# Speed of sound
def sound(U):
    rho_d_per, rho_m_per, rho_r_per, rhoU1, rhoU2, rhoU3, E_per, _rho_v,\
        _rho_c, T, _rho_d_0, _rho_m_0, _rho_r_0, E_0, _rho_v_0, T_0 = U

    rho_d_0, rho_r_0, rho_m_0 = Pos(_rho_d_0), Pos(_rho_r_0), Pos(_rho_m_0)
    rho_d = Pos(rho_d_per + rho_d_0)
    rho_m = Pos(rho_m_per + rho_m_0)
    rho_r = Pos(rho_r_per + rho_r_0)
    rho_v = Pos(_rho_v)
    rho_c = Pos(_rho_c)

    rho = rho_d + rho_m + rho_r
    p = R_d * rho_d * T + R_v * rho_v * T
    q_v, q_l = rho_v / rho_d, (rho_c + rho_r) / rho_d

    gamma_m = 1 + (R_d + R_v * q_v) / (c_vd + c_vv * q_v + c_l * q_l)

    return sqrt(gamma_m * p / rho)


# Maximal characteristic speed
def LamMax(A, B):
    rho_m_A = Pos(A[1] + Pos(A[11]))
    rho_r_A = Pos(A[2] + Pos(A[12]))
    rho_A = Pos(A[0] + Pos(A[10])) + rho_m_A + rho_r_A
    vr_A = V_r_func(rho_r_A, rho_m_A + rho_r_A)

    rho_m_B = Pos(B[1] + Pos(B[11]))
    rho_r_B = Pos(B[2] + Pos(B[12]))
    rho_B = Pos(B[0] + Pos(B[10])) + rho_m_B + rho_r_B
    vr_B = V_r_func(rho_r_B, rho_m_B + rho_r_B)

    lam_A = Abs(InnerProduct(A[3:6] / rho_A, n)) + Abs(vr_A * n[0]) + sound(A)
    lam_B = Abs(InnerProduct(B[3:6] / rho_B, n)) + Abs(vr_B * n[0]) + sound(B)
    return Max(lam_A, lam_B)


# Numerical Flux (local Lax-Friedrich)
def Fhatn(U):
    Ubnd = CF((U[:3],
               (1 - 2 * n[0]**2) * U[3] - 2 * n[0] * n[1] * U[4] - 2 * n[0] * n[2] * U[5],
               - 2 * n[1] * n[0] * U[3] + (1 - 2 * n[1]**2) * U[4] - 2 * n[1] * n[2] * U[5],
               - 2 * n[2] * n[0] * U[3] - 2 * n[2] * n[1] * U[4] + (1 - 2 * n[2]**2) * U[5],
               U[6:]))
    Uhat = U.Other(bnd=Ubnd)

    # Construct numerical flux
    _Fhat = 1 / 2 * (F(U) + F(Uhat)) * n
    _Fhat += 1 / 2 * LamMax(U, Uhat) * (U - Uhat)
    return _Fhat


def G(U):
    rho_d_per, rho_m_per, rho_r_per, rhoU1, rhoU2, rhoU3, E_per, _rho_v,\
        _rho_c, T, _rho_d_0, _rho_m_0, _rho_r_0, E_0, _rho_v_0, T_0 = U

    rho_d_0, rho_r_0, rho_m_0 = Pos(_rho_d_0), Pos(_rho_r_0), Pos(_rho_m_0)
    rho_0 = rho_d_0 + rho_m_0 + rho_r_0
    rho_d = Pos(rho_d_per + rho_d_0)
    rho_m = Pos(rho_m_per + rho_m_0)
    rho_r = Pos(rho_r_per + rho_r_0)
    rho = rho_d + rho_m + rho_r
    rho_per = rho - rho_0

    rho_v = Pos(_rho_v)
    rho_c = Pos(_rho_c)

    rho_vs = clausius_clapeyron(T) / (R_v * T)

    s_ev = (3.86e-3 - 9.41e-5 * (T - T_ref))
    s_ev *= (1 + 9.1 * rho_r**(3 / 16)) * (rho_vs - rho_v) * rho_r**(1 / 2)
    s_au = 0.001 * rho_c
    s_ac = 1.72 * rho_c * rho_r**(7 / 8)

    rain_sources = s_au + s_ac - s_ev

    G1 = 0
    G2 = rain_sources
    G3 = - rain_sources
    G41 = rho_per * g
    G42 = 0
    G43 = 0
    G5 = rhoU1 * g
    return CF((G1, G2, G3, G41, G42, G43, G5, 0, 0, 0, 0, 0, 0, 0, 0, 0))


# -------------------------------- INTEGRATORS --------------------------------
comp = {'realcompile': compile_flag, 'wait': wait_compile}

dX = dx(bonus_intorder=order)
dS_int = dx(skeleton=True, bonus_intorder=order)
dS_bnd = ds(skeleton=True, bonus_intorder=order)

a = BilinearForm(X, nonassemble=True)
a += - InnerProduct(F(U), Grad(V)).Compile(**comp) * dX
a += InnerProduct(Fhatn(U)[:7], (V - V.Other())[:7]).Compile(**comp) * dS_int
a += InnerProduct(Fhatn(U)[:7], V[:7]).Compile(**comp) * dS_bnd
a += InnerProduct(G(U), V).Compile(**comp) * dX


# --------------------------- ARTIFICIAL DIFFUSION ----------------------------
def add_diffusion(a, U, V):
    def avg(u0, u1):
        return (u0 + u1) * 0.5

    h = specialcf.mesh_size
    n = specialcf.normal(mesh.dim)
    _alpha = alpha * order**2 / h

    Ubnd = CF((U[:3],
               (1 - 2 * n[0]**2) * U[3] - 2 * n[0] * n[1] * U[4] - 2 * n[0] * n[2] * U[5],
               - 2 * n[1] * n[0] * U[3] + (1 - 2 * n[1]**2) * U[4] - 2 * n[1] * n[2] * U[5],
               - 2 * n[2] * n[0] * U[3] - 2 * n[2] * n[1] * U[4] + (1 - 2 * n[2]**2) * U[5],
               U[6:]))
    Uhat = U.Other(bnd=Ubnd)

    # Diffusion for rain density
    diff_r = magnitude_r / sqrt(h) * 0.5 * scale_diff
    u, u_oth = U[2], Uhat[2]
    grad_u, grad_u_oth = Grad(U)[2, :], Grad(Uhat)[2, :]
    v, v_oth = V[2], V.Other()[2]
    grad_v, grad_v_oth = Grad(V)[2, :], Grad(V.Other())[2, :]

    diff_vol = diff_r * InnerProduct(grad_u, grad_v)

    diff_int = -diff_r * InnerProduct(avg(grad_u, grad_u_oth) * n, v - v_oth)
    diff_int += -diff_r * InnerProduct(avg(grad_v, grad_v_oth) * n, u - u_oth)
    diff_int += diff_r * _alpha * InnerProduct(u - u_oth, v - v_oth)

    diff_bnd = -diff_r * InnerProduct(avg(grad_u, grad_u_oth) * n, v)
    diff_bnd += -diff_r * InnerProduct(grad_v * n, u - u_oth)
    diff_bnd += diff_r * _alpha * InnerProduct(u - u_oth, v)

    a += diff_vol.Compile(**comp) * dX
    a += diff_int.Compile(**comp) * dS_int
    a += diff_bnd.Compile(**comp) * dS_bnd

    # Diffusion for moist and dry densities
    diff_v = magnitude_v / sqrt(h) * 0.5 * scale_diff
    u, u_oth = U[:2], Uhat[:2]
    grad_u, grad_u_oth = Grad(U)[:2, :], Grad(Uhat)[:2, :]
    v, v_oth = V[:2], V.Other()[:2]
    grad_v, grad_v_oth = Grad(V)[:2, :], Grad(V.Other())[:2, :]

    diff_vol = diff_v * InnerProduct(grad_u, grad_v)

    diff_int = -diff_v * InnerProduct(avg(grad_u, grad_u_oth) * n, v - v_oth)
    diff_int += -diff_v * InnerProduct(avg(grad_v, grad_v_oth) * n, u - u_oth)
    diff_int += diff_v * _alpha * InnerProduct(u - u_oth, v - v_oth)

    diff_bnd = -diff_v * InnerProduct(avg(grad_u, grad_u_oth) * n, v)
    diff_bnd += -diff_v * InnerProduct(grad_v * n, u - u_oth)
    diff_bnd += diff_v * _alpha * InnerProduct(u - u_oth, v)

    a += diff_vol.Compile(**comp) * dX
    a += diff_int.Compile(**comp) * dS_int
    a += diff_bnd.Compile(**comp) * dS_bnd

    # Diffusion for momentum and energy density
    u, u_oth = U[3:7], Uhat[3:7]
    grad_u, grad_u_oth = Grad(U)[3:7, :], Grad(Uhat)[3:7, :]
    v, v_oth = V[3:7], V.Other()[3:7]
    grad_v, grad_v_oth = Grad(V)[3:7, :], Grad(V.Other())[3:7, :]

    diff_vol = diff_v * InnerProduct(grad_u, grad_v)

    diff_int = -diff_v * InnerProduct(avg(grad_u, grad_u_oth) * n, v - v_oth)
    diff_int += -diff_v * InnerProduct(avg(grad_v, grad_v_oth) * n, u - u_oth)
    diff_int += diff_v * _alpha * InnerProduct(u - u_oth, v - v_oth)

    diff_bnd = -diff_v * InnerProduct(avg(grad_u, grad_u_oth) * n, v)
    diff_bnd += -diff_v * InnerProduct(grad_v * n, u - u_oth)
    diff_bnd += diff_v * _alpha * InnerProduct(u - u_oth, v)

    a += diff_vol.Compile(**comp) * dX
    a += diff_int.Compile(**comp) * dS_int
    a += diff_bnd.Compile(**comp) * dS_bnd

    return None


if artificial_diffusion:
    X_H1 = H1(mesh, order=1)
    magnitude_r, magnitude_v = GridFunction(X_H1), GridFunction(X_H1)
    gf_elwise_const = GridFunction(X0)
    magnitude_r.vec.data[:] = 0.0
    magnitude_v.vec.data[:] = 0.0
    add_diffusion(a, U, V)

gf_vr = V_r_func(gf_rho_r, gf_rho_m + gf_rho_r).Compile()


# --------------------------- TEMPERATURE PROBLEM ----------------------------
def energy_problem(U_irs):
    rho_v, rho_c, T = U_irs

    _rho_vs = clausius_clapeyron(T) / (R_v * T)
    _rho_v = Min(_rho_vs, gf_rho_m)
    _rho_c = gf_rho_m - _rho_v

    _rhoe = c_vd * gf_rho_d + c_vv * _rho_v + c_l * (_rho_c + gf_rho_r)
    _rhoe *= (T - T_ref)
    _rhoe += _rho_v * (L_ref - R_v * T_ref)

    f1 = rho_v - _rho_v
    f2 = rho_c - _rho_c
    f3 = _rhoe - gf_rhoe

    return CF((1e7 * f1, 1e7 * f2, f3))


dx_irs = dx(intrules=IRS.components[0].GetIntegrationRules())
u_irs, v_irs = IRS.TnT()

temp_problem = NewtonCF((energy_problem(u_irs)).Compile(**comp),
                        gf_X2, tol=newton_tol, maxiter=newton_maxit)

f_T = LinearForm(X2)
f_T += InnerProduct(temp_problem, X2.TestFunction()) * dx_irs

inv_mX2 = X2.Mass(1).Inverse()


def solve_temperature(vec):
    gf_X2.vec.data = vec[X1.ndof:X1.ndof + X2.ndof]
    f_T.Assemble()
    vec[X1.ndof:X1.ndof + X2.ndof].data = inv_mX2 * f_T.vec
    return None


# -------------------------- HYDROSTATIC BASE STATE ---------------------------
def initial_constraint_residual(p, U):
    rho_d, rho_v, T = U

    rho_vs = clausius_clapeyron(T) / (R_v * T)

    res1 = p - T * (rho_d * R_d + rho_v * R_v)
    res2 = theta_d0 - T * (p_ref / p)**(R_d / c_pd)
    res3 = rho_vs * (rho_d + rho_v / eps) * humidity_rel0
    res3 -= rho_v * (rho_d + rho_vs / eps)

    return CF((res1, res2, 1000 * res3))


(p_1d, U_1d), (q_1d, V_1d) = X_1d.TnT()

a_1d = BilinearForm(X_1d)
a_1d += (Grad(p_1d) + g * (U_1d[0] + U_1d[1])) * q_1d * dx
a_1d += InnerProduct(initial_constraint_residual(p_1d, U_1d), V_1d) * dx

gfu_1d.vec[:] = 0.0
gfu_1d_2.vec[:] = 0.0

gfu_1d.components[0].Set(p_surf)
gfu_1d_2.components[1].Set((1, 0.001, T_surf))
gfu_1d.vec.data += gfu_1d_2.vec

with TaskManager():
    Newton(a_1d, gfu_1d, maxerr=1e-10)

gf_rho_d_h, gf_rho_v_h, gf_T_h = gfu_1d.components[1]

gf_X.Set((0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          gf_rho_d_h,
          gf_rho_v_h + 0,
          0,
          (gf_rho_d_h * c_vd + c_vv * gf_rho_v_h + c_l * 0) * (gf_T_h - T_ref)
          + gf_rho_v_h * (L_ref - R_v * T_ref),
          gf_rho_v_h,
          gf_T_h))


del mesh1d, X1_1d, X2_1d, p_1d, U_1d, q_1d, V_1d, gfu_1d_2, a_1d


# ----------------------------- INITIAL CONDITION -----------------------------
if restart_flag is False:
    def initial_problem_res(U):
        rho_d, rho_v, T = U

        pre_h = (gf_rho_d_h * R_d + gf_rho_v_h * R_v) * gf_T_h

        rho_vs = clausius_clapeyron(T) / (R_v * T)
        pre = (rho_d * R_d + rho_v * R_v) * T

        res1 = rho_vs * (rho_d + rho_v / eps) * H_init
        res1 -= rho_v * (rho_d + rho_vs / eps)
        res2 = theta_d0 - T * (p_ref / pre_h)**(R_d / c_pd)
        res3 = pre_h - pre
        return CF((30 * res1, res2, res3))

    initial_problem = NewtonCF(initial_problem_res(u_irs).Compile(),
                               gfu_1d.components[1], rtol=newton_tol,
                               maxiter=newton_maxit)

    f_irs_init = LinearForm(IRS)
    f_irs_init += initial_problem * v_irs * dx_irs

    mass_irs = BilinearForm(IRS, symmetric=True, diagonal=True)
    mass_irs += (u_irs * v_irs).Compile() * dx_irs

    with TaskManager():
        mass_irs.Assemble()
        f_irs_init.Assemble()
        gf_IRS.vec.data = mass_irs.mat.Inverse() * f_irs_init.vec

    _init_rho_d = gf_IRS[0]
    _init_rho_v = gf_IRS[1]
    _init_T = gf_IRS[2]
    _init_rho_c = 0
    _init_rho_m = _init_rho_v + _init_rho_c
    _init_rho_r = 0

    _init_rhoe = c_vd * _init_rho_d + c_vv * _init_rho_v
    _init_rhoe += c_l * (_init_rho_c + _init_rho_r)
    _init_rhoe *= (_init_T - T_ref)
    _init_rhoe += _init_rho_v * (L_ref - R_v * T_ref)

    _init = CF((_init_rho_d - gf_rho_d_0,
                _init_rho_m - gf_rho_m_0,
                _init_rho_r - gf_rho_r_0,
                0, 0, 0,
                _init_rhoe - gf_rhoe_0,
                _init_rho_v,
                _init_rho_c,
                _init_T,
                0, 0, 0, 0, 0, 0))

    f_init = LinearForm(X)
    f_init += InnerProduct(_init, X.TestFunction()).Compile() * dx_irs

    with TaskManager():
        f_init.Assemble()

    del _init_T, _init_rho_d, _init_rho_m, _init_rho_v, _init_rho_c, _init_rhoe


# ------------------------------- VISUALISATION -------------------------------
Draw(gf_vel, mesh, 'vel')
Draw(gf_rhoe, mesh, 'rhoe')
Draw(gf_rho_v, mesh, 'rho_v')
Draw(gf_rho_c, mesh, 'rho_c')
Draw(gf_T, mesh, 'T')
Draw(gf_rho_d, mesh, 'rho_d')
Draw(gf_rho_m, mesh, 'rho_m')
Draw(gf_E, mesh, 'rhoE')
Draw(gf_rho_r, mesh, 'rho_r')


# --------------------------- MEASURE PRECIPITATION ---------------------------
ba_els_left = BitArray(mesh.ne)
ba_els_left.Clear()
vertex_nrs_left = set([v.nr for el in mesh.Elements(BND) if el.mat == 'left'
                       for v in el.vertices])
for el in mesh.Elements(VOL):
    for v in el.vertices:
        if v.nr in vertex_nrs_left:
            ba_els_left.Set(el.nr)


_X_precip = L2(mesh, order=order)
gfu_precip_loc, gfu_precip = GridFunction(_X_precip), GridFunction(_X_precip)
gfu_precip.vec.data[:] = 0.0


def collect_rainfall():
    gfu_precip_loc.Set(gf_rho_r * (gf_vel[0] - gf_vr),
                       definedonelements=ba_els_left)
    gfu_precip.vec.data += dt * gfu_precip_loc.vec
    return None


# ---------------------------------- OUTPUT -----------------------------------
try:
    data_dir = os.environ['DATA']
except KeyError:
    print('DATA environment variable does not exist')
    data_dir = '..'
if vtk_flag or pickle_flag:
    print(f'data will be saved in the directory {data_dir}')

comp_dir_name = os.getcwd().split('/')[-1]
pickle_dir_abs = data_dir + '/' + comp_dir_name + '/' + pickle_dir

if pickle_flag:
    if not os.path.isdir(pickle_dir_abs):
        os.makedirs(pickle_dir_abs)

    def do_pickle(it):
        filename = pickle_dir_abs + '/' + pickle_name + f'_step{it}.dat'
        pickler = pickle.Pickler(open(filename, 'wb'))
        pickler.dump(gf_X.vec)
        return None


if vtk_flag:
    vtk_dir_abs = data_dir + '/' + comp_dir_name + '/' + vtk_dir
    if not os.path.isdir(vtk_dir_abs):
        os.makedirs(vtk_dir_abs)

    if restart_flag:
        vtk_name += 'restarted'

    vtk = VTKOutput(ma=mesh,
                    coefs=[gf_rho_d, gf_rho_v, gf_rho_c, gf_rho_r, gf_vel,
                           gf_E, gf_T, gf_vr, magnitude_r, magnitude_v,
                           gfu_precip],
                    names=['rho_d', 'rho_v', 'rho_c', 'rho_r', 'vel', 'rhoE',
                           'T', 'vr', 'mag_r', 'mag_v', 'precip_bot'],
                    filename=vtk_dir_abs + '/' + vtk_name,
                    subdivision=vtk_subdiv,
                    order=vtk_order,
                    floatsize='single',
                    legacy=False)


# ------------------------------- TIME STEPPING -------------------------------
def _sqr(vec):
    return InnerProduct(vec, vec)


res, U0, U1, U2, U3 = [gf_X.vec.CreateVector() for i in range(5)]


with TaskManager():

    inv_m = X1.Mass(1).Inverse()

    if restart_flag:
        _file = pickle_dir_abs + '/' + pickle_name + f'_step{restart_step}.dat'
        _unpickler = pickle.Unpickler(open(_file, 'rb'))
        gf_X.vec.data = _unpickler.load()
        del _file, _unpickler
        start_step = restart_step + 1
        Redraw()
    else:
        # Store hydrostatic base state
        if pickle_flag:
            do_pickle(-1)

        res.data = X.Mass(1).Inverse() * f_init.vec
        gf_X.vec.data[:X1.ndof + X2.ndof] += res[:X1.ndof + X2.ndof]

        del f_init

        if pickle_flag:
            do_pickle(0)
        if vtk_flag:
            vtk.Do(time=0.0)

        start_step = 1

    for it in range(start_step, int(t_end / dt) + 1):
        # Explicit Euler
        # a.Apply(gf_X.vec, res)
        # gf_X.vec[:X1.ndof].data -= dt * inv_m * res[:X1.ndof]
        # solve_temperature(gf_X.vec)

        # SSPRK(4,3)
        U0.data = gf_X.vec

        a.Apply(U0, res)
        gf_X.vec[:X1.ndof].data = U0[:X1.ndof] - dt / 2 * inv_m * res[:X1.ndof]
        solve_temperature(gf_X.vec)
        U1.data = gf_X.vec

        a.Apply(U1, res)
        gf_X.vec[:X1.ndof].data = U1[:X1.ndof] - dt / 2 * inv_m * res[:X1.ndof]
        solve_temperature(gf_X.vec)
        U2.data = gf_X.vec

        a.Apply(U2, res)
        gf_X.vec[:X1.ndof].data = 2 / 3 * U0[:X1.ndof] + 1 / 3 * U2[:X1.ndof]
        gf_X.vec[:X1.ndof].data += - dt / 6 * inv_m * res[:X1.ndof]
        solve_temperature(gf_X.vec)
        U3.data = gf_X.vec

        a.Apply(U3, res)
        gf_X.vec[:X1.ndof].data = U3[:X1.ndof] - dt / 2 * inv_m * res[:X1.ndof]
        solve_temperature(gf_X.vec)

        if artificial_diffusion:
            norm_vel = Integrate(_sqr(gf_vel[0] - gf_vr) + _sqr(gf_vel[1]),
                                 mesh, order=2 * order, element_wise=True)
            gf_elwise_const.vec.FV().NumPy()[:] = norm_vel
            magnitude_r.Set(sqrt(gf_elwise_const))
            norm_vel = Integrate(_sqr(gf_vel), mesh, order=2 * order,
                                 element_wise=True)
            gf_elwise_const.vec.FV().NumPy()[:] = norm_vel
            magnitude_v.Set(sqrt(gf_elwise_const))

        collect_rainfall()

        Redraw(blocking=False)
        if pickle_flag and it % pickle_freq == 0:
            do_pickle(it)
        if vtk_flag and it % vtk_freq == 0:
            vtk.Do(time=dt * it)
        if it % chech_nan_freq == 0:
            if isnan(Norm(gf_X.vec)):
                print(f'Aborting as NaNs detected!, t={it * dt}')
                break

        print(f't={dt * it:11.7f}', end='\r')

    if pickle_flag:
        do_pickle(int(t_end / dt))


# ------------------------------ POST-PROCESSING ------------------------------
time_total = wall_time.time() - time_start

print('\n----- Total time: {:02.0f}:{:02.0f}:{:02.0f}:{:06.3f} -----\n'.format(
      time_total // (24 * 60**2), time_total % (24 * 60**2) // 60**2,
      time_total % 3600 // 60, time_total % 60))
