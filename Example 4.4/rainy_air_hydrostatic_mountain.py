'''
    Hydrostatic atmosphere with mountain terrain, inspired  based on
    [1]. The microphysics parametrisation is taken from the COSMO model
    [2].

    Note: Due to implementational issues, the x and y axis are swapped.

    Literature
    ----------
    [1] G. Zängl. ‘Extending the Numerical Stability Limit of Terrain-
        Following Coordinate Models over Steep Slopes’. In: Mon. Wea.
        Rev. 140.11 (2012), pp. 3722–3733.
    [2] G. Doms et al. COSMO-Model Version 6.00: A Description of the
        Nonhydrostatic Regional COSMO- Model - Part II: Physical
        Parametrizations. Tech. rep. COSMO Consortium for Small-Scale
        Modelling, 2021.
'''
# ------------------------------ LOAD LIBRARIES -------------------------------
from netgen.occ import *
from netgen.meshing import IdentificationType
from ngsolve import *
from ngsolve.meshes import Make1DMesh
from ngsolve.comp import IntegrationRuleSpace
from ngsolve.fem import NewtonCF
from ngsolve.solvers import Newton
from time import time
from math import isnan, gamma
import os
import pickle
import argparse

SetHeapSize(10000000)
SetNumThreads(6)
ngsglobals.symbolic_integrator_uses_diff = True
time_start = time()


# --------------------------- MPI / SHARED MEMEORY ----------------------------
comm = mpi_world
rank = comm.rank

if comm.size == 1:
    print('Running shared memery paralell')
elif rank == 0:
    print(f'Running MPI parallel on {comm.size} ranks')


# ------------------------- COMMAND-LINE PARAMETERS ---------------------------
parser = argparse.ArgumentParser(description='Hydrostatic atmosphere with mointain',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-hmax', '--mesh_size', type=float, default=1000, help='Mesh size')
parser.add_argument('-qm', '--quads', type=int, default=0, help='Triangular mesh if 0 or quad mesh if 1')
parser.add_argument('-o', '--order', type=int, default=2, help='Order of finite element space')
parser.add_argument('-dt', '--time_step', type=float, default=0.2, help='Time step')
parser.add_argument('-ad', '--diffusion', default=0, type=int, help='Add Péclet scaled diffusion if 1')
parser.add_argument('-dp', '--diffusion_par', default=0.05, type=float, help='Artificial diffusion parameter')
options = vars(parser.parse_args())
print(options)


# -------------------------------- PARAMETERS ---------------------------------
h_max = options['mesh_size']
order = options['order']
quad_mesh = bool(options['quads'])

t_end = 6 * 60 * 60
dt = options['time_step']

wait_compile = False
compile_flag = False

newton_tol = 1e-9
newton_maxit = 10

artificial_diffusion = bool(options['diffusion'])
scale_diff = options['diffusion_par']
if artificial_diffusion is False:
    scale_diff = 0.0
alpha = 10

blend_d = 15e3
blend_alpha = 0.1

pickle_flag = False
pickle_dir = 'pickle_output'
pickle_name = f'hydrostatic_mointain_rain_dg_ssprk43_pert_artdiff{scale_diff}'
pickle_name += f'quads{int(quad_mesh)}h{h_max}k{order}dt{dt}'
pickle_freq = int(100 / dt)

vtk_flag = False
vtk_dir = 'vtk_output'
vtk_name = pickle_name
vtk_subdiv = 0
vtk_order = max(1, min(2, order))
vtk_freq = int(300 / dt)

restart_flag = False
restart_step = 0

chech_nan_freq = ceil(10 / dt)


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

domain_h = 40e3             # Domain height
domain_w = 35e3             # Domain width


def height_func(x, x_c=domain_w / 2, h=7000, a=2000):
    h1 = h * exp(- (x - x_c)**2 / a**2)
    if h1 < 1e-13:
        h1 = 0
    return h1


H_scal = 1e5
T_sl = 288.15
T_str = 213.15
T_0_cf = T_str + (T_sl - T_str) * exp(- x / H_scal)


# ----------------------------------- MESH ------------------------------------
if rank == 0:
    _t1 = time()
    _N = 200
    pts = [(height_func(x), x) for x in [domain_w * i / _N
                                         for i in reversed(range(_N + 1))]]
    bottom = SplineInterpolation(pts)
    pts1 = [gp_Pnt2d(*p) for p in [(0, 0), (domain_h, 0), (domain_h, domain_w),
                                   (0, domain_w)]]
    right = Segment(pts1[0], pts1[1])
    top = Segment(pts1[1], pts1[2])
    left = Segment(pts1[2], pts1[3])
    wire = Wire([right.Edge(), top.Edge(), left.Edge(), bottom.Edge()])
    f = Face(wire)
    f.edges.Min(Y).Identify(f.edges.Max(Y), "periodic",
                            IdentificationType.PERIODIC)
    geo = OCCGeometry(f, dim=2)
    ngmesh = geo.GenerateMesh(maxh=h_max, quad_dominated=quad_mesh,
                              curvaturesafety=1).Distribute(comm)
    print(f'Time for serial meshing: {time() - _t1}')
else:
    ngmesh = netgen.meshing.Mesh.Receive(comm)
mesh = Mesh(ngmesh)


# --------------------------------- FE-SPACE ----------------------------------
X0 = L2(mesh, order=0)
X1 = L2(mesh, order=order)**6
X2 = L2(mesh, order=order)**3
X3 = L2(mesh, order=order)**9
X = L2(mesh, order=order)**16
IRS = IntegrationRuleSpace(mesh, order=order)**3

print(f'proc {rank}/{comm.size} has {X1.ndof}/{comm.Sum(X1.ndof)} dofs of X1')

gf_X = GridFunction(X)
gf_rho_d_per, gf_rho_m_per, gf_rho_r_per, gf_rhoU1, gf_rhoU2, gf_E_per, \
    gf_rho_v, gf_rho_c, gf_T, gf_rho_d_0, gf_rho_m_0, gf_rho_r_0, \
    gf_rhoe_0, gf_rho_v_0, gf_rho_c_0, gf_T_0 = gf_X.components
gf_rho_d = gf_rho_d_per + gf_rho_d_0
gf_rho_m = gf_rho_m_per + gf_rho_m_0
gf_rho_r = gf_rho_r_per + gf_rho_r_0
gf_rho = gf_rho_d + gf_rho_m + gf_rho_r
gf_rhovel = CF(gf_X.components[3:5])
gf_vel = gf_rhovel / gf_rho
gf_E = gf_E_per + gf_rhoe_0
gf_rhoe = gf_E - (gf_rhoU1**2 + gf_rhoU2**2) / (2 * gf_rho)

gf_X2 = GridFunction(X2)
gf_IRS = GridFunction(IRS)

mesh1d = Make1DMesh(n=int(ceil(domain_h / h_max)),
                    mapping=lambda x: x * domain_h)

X1_1d = H1(mesh1d, order=order + 1, dirichlet='left')
X2_1d = L2(mesh1d, order=order)
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
    rho_d_per, rho_m_per, rho_r_per, rhoU1, rhoU2, E_per, _rho_v, _rho_c, \
        T, _rho_d_0, _rho_m_0, _rho_r_0, E_0, _rho_v_0, _rho_c_0, T_0 = U

    rho_d_0, rho_r_0, rho_m_0 = Pos(_rho_d_0), Pos(_rho_r_0), Pos(_rho_m_0)
    rho_d = Pos(rho_d_per + rho_d_0)
    rho_m = Pos(rho_m_per + rho_m_0)
    rho_r = Pos(rho_r_per + rho_r_0)
    rho_v = Pos(_rho_v)
    rho_v_0 = Pos(_rho_v_0)
    rho_w = rho_m + rho_r
    rho = rho_d + rho_m + rho_r

    rhovel = U[3:5]
    vel = rhovel / rho
    v_r_vec = CoefficientFunction((V_r_func(rho_r, rho_w), 0))

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
    Fz = CF((0, 0))

    return CF((F1, F2, F3, F4, F5, Fz, Fz, Fz, Fz, Fz, Fz, Fz, Fz, Fz, Fz),
              dims=(16, 2))


# Speed of sound
def sound(U):
    rho_d_per, rho_m_per, rho_r_per, rhoU1, rhoU2, E_per, _rho_v, _rho_c, \
        T, _rho_d_0, _rho_m_0, _rho_r_0, E_0, _rho_v_0, _rho_c_0, T_0 = U

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
    rho_m_A = Pos(A[1] + Pos(A[10]))
    rho_r_A = Pos(A[2] + Pos(A[11]))
    rho_A = Pos(A[0] + Pos(A[9])) + rho_m_A + rho_r_A
    vr_A = V_r_func(rho_r_A, rho_m_A + rho_r_A)

    rho_m_B = Pos(B[1] + Pos(B[10]))
    rho_r_B = Pos(B[2] + Pos(B[11]))
    rho_B = Pos(B[0] + Pos(B[9])) + rho_m_B + rho_r_B
    vr_B = V_r_func(rho_r_B, rho_m_B + rho_r_B)

    lam_A = Abs(InnerProduct(A[3:5] / rho_A, n)) + Abs(vr_A * n[0]) + sound(A)
    lam_B = Abs(InnerProduct(B[3:5] / rho_B, n)) + Abs(vr_B * n[0]) + sound(B)
    return Max(lam_A, lam_B)


# Numerical Flux (local Lax-Friedrich)
def Fhatn(U):
    Ubnd = CF((U[:3],
               (1 - 2 * n[0]**2) * U[3] - 2 * n[0] * n[1] * U[4],
               -2 * n[0] * n[1] * U[3] + (1 - 2 * n[1]**2) * U[4],
               U[5:]))
    Uhat = U.Other(bnd=Ubnd)

    # Construct numerical flux
    _Fhat = 1 / 2 * (F(U) + F(Uhat)) * n
    _Fhat += 1 / 2 * LamMax(U, Uhat) * (U - Uhat)
    return _Fhat


def G(U):
    rho_d_per, rho_m_per, rho_r_per, rhoU1, rhoU2, E_per, _rho_v, _rho_c, \
        T, _rho_d_0, _rho_m_0, _rho_r_0, E_0, _rho_v_0, _rho_c_0, T_0 = U

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
    G5 = rhoU1 * g
    return CF((G1, G2, G3, G41, G42, G5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))


# -------------------------------- INTEGRATORS --------------------------------
comp = {'realcompile': compile_flag, 'wait': wait_compile}

dX = dx(bonus_intorder=order)
dS_int = dx(skeleton=True, bonus_intorder=order)
dS_bnd = ds(skeleton=True, bonus_intorder=order)

a = BilinearForm(X, nonassemble=True)
a += - InnerProduct(F(U), Grad(V)).Compile(**comp) * dX
a += InnerProduct(Fhatn(U)[:6], (V - V.Other())[:6]).Compile(**comp) * dS_int
a += InnerProduct(Fhatn(U)[:6], V[:6]).Compile(**comp) * dS_bnd
a += InnerProduct(G(U), V).Compile(**comp) * dX


# --------------------------- ARTIFICIAL DIFFUSION ----------------------------
def add_diffusion(a, U, V):
    def avg(u0, u1):
        return (u0 + u1) * 0.5

    h = specialcf.mesh_size
    n = specialcf.normal(mesh.dim)
    _alpha = alpha * order**2 / h

    Ubnd = CF((U[:3],
               (1 - 2 * n[0]**2) * U[3] - 2 * n[0] * n[1] * U[4],
               -2 * n[0] * n[1] * U[3] + (1 - 2 * n[1]**2) * U[4],
               U[5:]))
    Uhat = U.Other(bnd=Ubnd)

    # Diffusion for rain density
    diff_r = magnitude_r * 0.5 * scale_diff
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
    diff_v = magnitude_v * 0.5 * scale_diff
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
    u, u_oth = U[3:6], Uhat[3:6]
    grad_u, grad_u_oth = Grad(U)[3:6, :], Grad(Uhat)[3:6, :]
    v, v_oth = V[3:6], V.Other()[3:6]
    grad_v, grad_v_oth = Grad(V)[3:6, :], Grad(V.Other())[3:6, :]

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


X_H1 = H1(mesh, order=1)
magnitude_r, magnitude_v = GridFunction(X_H1), GridFunction(X_H1)
magnitude_r.vec.data[:], magnitude_v.vec.data[:] = 0.0, 0.0

if artificial_diffusion:
    gf_elwise_const = GridFunction(X0)
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


# ------------------------------ DAMPING LAYER --------------------------------
blend_w = domain_h - blend_d
blend_cf = blend_alpha / 2 * (1 - cos(pi * (x - blend_d) / blend_w))
blend_cf = IfPos(x - blend_d, blend_cf, 0)

gfu_hydro = CF((0, 0, 0, 0, 0, 0, gf_rho_v_0, gf_rho_c_0, gf_T_0))

f_blend = LinearForm(X3)
f_blend += InnerProduct((1 - blend_cf) * gf_X[:9] + blend_cf * gfu_hydro,
                        X3.TestFunction()).Compile(**comp) * dx

inv_mX3 = X3.Mass(1).Inverse()


def apply_blending():
    f_blend.Assemble()
    gf_X.vec.data[:X3.ndof] = inv_mX3 * f_blend.vec
    return None


# -------------------------- HYDROSTATIC BASE STATE ---------------------------
(p_1d, U_1d), (q_1d, V_1d) = X_1d.TnT()

rho_vs_0 = clausius_clapeyron(T_0_cf) / (R_v * T_0_cf)

a_1d = BilinearForm(X_1d)
a_1d += (Grad(p_1d) + g * (U_1d + rho_vs_0)) * q_1d * dx
a_1d += (p_1d - T_0_cf * (U_1d * R_d + rho_vs_0 * R_v)) * V_1d * dx

# Solve 1-d problem in serial on rank 0
if rank == 0:
    gfu_1d.vec[:] = 0.0
    gfu_1d_2.vec[:] = 0.0

    gfu_1d.components[0].Set(p_ref)
    gfu_1d_2.components[1].Set(1.1)
    gfu_1d.vec.data += gfu_1d_2.vec

    Newton(a_1d, gfu_1d, maxerr=1e-11)

if comm.size > 1:
    comm.mpi4py.Bcast(gfu_1d.vec.FV().NumPy(), root=0)

gf_rho_d_h = gfu_1d.components[1]

gf_X.Set((0, 0, 0, 0, 0, 0,
          rho_vs_0 + 0, 0, T_0_cf + 0,
          gf_rho_d_h,
          rho_vs_0 + 0,
          0,
          (gf_rho_d_h * c_vd + c_vv * rho_vs_0 + c_l * 0) * (T_0_cf - T_ref)
          + rho_vs_0 * (L_ref - R_v * T_ref),
          rho_vs_0,
          0,
          T_0_cf))

del mesh1d, X1_1d, X2_1d, p_1d, U_1d, q_1d, V_1d, gfu_1d_2, a_1d


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

rho_vs_cf = clausius_clapeyron(T_0_cf) / (R_v * T_0_cf)
gf_humid = gf_rho_v / rho_vs_cf * (eps * gf_rho_d + rho_vs_cf) / (eps * gf_rho_d + gf_rho_v)
Draw(gf_humid - 1, mesh, 'humid')

Draw(gf_X, mesh, 'gf')


# ---------------------------------- OUTPUT -----------------------------------
try:
    data_dir = os.environ['DATA']
except KeyError:
    print('DATA environment variable does not exist')
    data_dir = '..'
if (vtk_flag or pickle_flag) and rank == 0:
    print(f'data will be saved in the directory {data_dir}')

comp_dir_name = os.getcwd().split('/')[-1]
pickle_dir_abs = f'{data_dir}/{comp_dir_name}/{pickle_dir}'

if pickle_flag:
    if not os.path.isdir(pickle_dir_abs) and rank == 0:
        os.makedirs(pickle_dir_abs)
    comm.Barrier()

    def do_pickle(it):
        filename = f'{pickle_dir_abs}/{pickle_name}_step{it}_rank{rank}.dat'
        pickler = pickle.Pickler(open(filename, 'wb'))
        pickler.dump(gf_X.vec)
        return None


if vtk_flag:
    vtk_dir_abs = f'{data_dir}/{comp_dir_name}/{vtk_dir}'
    if not os.path.isdir(vtk_dir_abs) and rank == 0:
        os.makedirs(vtk_dir_abs)
    comm.Barrier()

    if restart_flag:
        vtk_name += 'restarted'

    vtk = VTKOutput(ma=mesh,
                    coefs=[gf_rho_d, gf_rho_m, gf_rho_r, gf_rho_v, gf_rho_c,
                           gf_vel, gf_E, gf_T, gf_vr],
                    names=['rho_d', 'rho_m', 'rho_r', 'rho_v', 'rho_c',
                           'vel', 'E', 'T', 'vr'],
                    filename=vtk_dir_abs + '/' + vtk_name,
                    subdivision=vtk_subdiv,
                    order=vtk_order,
                    floatsize='single',
                    legacy=False)


# ------------------------------- TIME STEPPING -------------------------------
if comm.size == 1:
    tm = TaskManager()
    tm.__enter__()


def _sqr(vec):
    return InnerProduct(vec, vec)


res, U0, U1, U2, U3 = [gf_X.vec.CreateVector() for i in range(5)]

inv_m = X1.Mass(1).Inverse()

if restart_flag:
    _file = f'{pickle_dir_abs}/{pickle_name}_step{restart_step}_rank{rank}.dat'
    _unpickler = pickle.Unpickler(open(_file, 'rb'))
    gf_X.vec.data = _unpickler.load()
    del _file, _unpickler
    start_step = restart_step + 1
    Redraw()
else:
    # Store hydrostatic base state
    if pickle_flag:
        do_pickle(-1)

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

    apply_blending()

    if comm.size == 1:
        Redraw(blocking=False)
    if pickle_flag and it % pickle_freq == 0:
        do_pickle(it)
    if vtk_flag and it % vtk_freq == 0:
        vtk.Do(time=dt * it)
    if rank == 1 or comm.size == 1:
        print(f't={dt * it:11.7f}', end='\r')

    if it % chech_nan_freq == 0:
        if isnan(Norm(gf_X.vec)):
            print(f'Aborting as NaNs detected!, t={it * dt}')
            break

if pickle_flag:
    do_pickle(int(t_end / dt))

if comm.size == 1:
    tm.__exit__(None, None, None)


# ------------------------------ POST-PROCESSING ------------------------------
time_total = time() - time_start

if rank == 1 or comm.size == 1:
    print('\n---- Total time: {:02.0f}:{:02.0f}:{:02.0f}:{:06.3f} ----'.format(
          time_total // (24 * 60**2), time_total % (24 * 60**2) // 60**2,
          time_total % 3600 // 60, time_total % 60))
