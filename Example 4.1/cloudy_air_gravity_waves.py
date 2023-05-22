'''
    Inertia Gravity waves in a saturated atmosphere. See [1] Sec 5.2.
    Computation based on the perturbation formulation.

    Note: Due to implementational issues, the x and y axis are swapped.

    Literature
    ----------
    [1] T. M. Bendall et al. ‘A compatible finite-element discretisation
        for the moist compressible Euler equations’. In: Quart. J. Roy.
        Meteorol. Soc. 146.732 (2020), pp. 3187–3205. doi: 10.1002/qj.3841.
'''
# ------------------------------ LOAD LIBRARIES -------------------------------
from netgen.geom2d import SplineGeometry
from ngsolve.meshes import Make1DMesh, MakeStructured2DMesh
from ngsolve import *
from ngsolve.comp import IntegrationRuleSpace
from ngsolve.fem import NewtonCF
from ngsolve.solvers import Newton
from time import time
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
parser = argparse.ArgumentParser(description='Inertia gravity waves in a over-saturated atmosphere',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-hmax', '--mesh_size', default=1000, type=float, help='Mesh diameter')
parser.add_argument('-qm', '--quads', type=int, default=1, help='Quad mesh if 1 or triangular mesh if 0')
parser.add_argument('-sm', '--struc', type=int, default=1, help='Structured mesh if 1 or unstrucred mesh if 0')
parser.add_argument('-o', '--order', default=3, type=int, help='Polynomial order of the finite element space')
parser.add_argument('-dt', '--time_step', default=0.3, type=float, help='Time step for SSPRK43 scheme')
parser.add_argument('-ad', '--diffusion', default=0, type=int, help='Add articficial Péclet scaled diffusion if 1')
parser.add_argument('-dp', '--diffusion_par', default=0.0001, type=float, help='Artificial diffusion parameter')
options = vars(parser.parse_args())
if rank == 0:
    print(options)


# -------------------------------- PARAMETERS ---------------------------------
h_max = options['mesh_size']
order = options['order']
quad_mesh = bool(options['quads'])
struc_mesh = bool(options['struc'])

t_end = 3600
dt = options['time_step']

wait_compile = False
compile_flag = False

newton_tol = 1e-9
newton_maxit = 10

artificial_diffusion = bool(options['diffusion'])
scale_diff = options['diffusion_par']
if artificial_diffusion is False:
    scale_diff = 0
alpha = 10

pickle_flag = False
pickle_dir = 'pickle_output'
pickle_name = f'cloudy_gravity_waves_dg_ssprk43_artdiff{scale_diff}'
pickle_name += f'quads{int(quad_mesh)}struc{int(struc_mesh)}'
pickle_name += f'h{h_max}k{order}dt{dt}'
pickle_freq = int(t_end / dt / 30)

vtk_flag = False
vtk_dir = 'vtk_output'
vtk_name = pickle_name
vtk_subdiv = 0
vtk_order = min(2, max(1, order))
vtk_freq = int(t_end / dt / 30)

restart_flag = False
restart_step = 0


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

g = 9.81                    # Acceleration due to gravity

Theta_0 = 300               # Initial (wet eqiv. potent.) temperature
N2 = 1e-4                   # Initial profile parameter
theta_e_barx = Theta_0 * exp(N2 * x / g)  # Switch x and y-directions!!!
q_t = 0.02                  # Total water mixing ratio

DeltaTheta = 0.01           # Perturbation wet eqiv. potent. temperature
_a = 5e3                    # Perturbation parameter

domain_h = 10000            # Domain height
domain_w = 300000           # Domain width

vel_x_init = 20             # Initial horizontal velocity


# ----------------------------------- MESH -----------------------------------
if rank == 0:
    _t1 = time()
    if struc_mesh:
        ngmesh = MakeStructured2DMesh(
            quads=quad_mesh, periodic_y=True,
            nx=int(ceil(domain_h / h_max)),
            ny=int(ceil(domain_w / h_max)),
            mapping=lambda x, y: (domain_h * x, domain_w * y)).ngmesh
    else:
        geo = SplineGeometry()
        pnts = [(0, 0), (domain_h, 0), (domain_h, domain_w), (0, domain_w)]
        pnums = [geo.AppendPoint(*p) for p in pnts]
        bot = geo.Append(["line", pnums[0], pnums[1]])
        geo.Append(["line", pnums[1], pnums[2]])
        geo.Append(["line", pnums[3], pnums[2]],
                   leftdomain=0, rightdomain=1, copy=bot)
        geo.Append(["line", pnums[3], pnums[0]])
        ngmesh = geo.GenerateMesh(maxh=h_max, quad_dominated=quad_mesh)
    print(f'time for serial meshing: {time() - _t1}')
    ngmesh = ngmesh.Distribute(comm)
else:
    ngmesh = netgen.meshing.Mesh.Receive(comm)
mesh = Mesh(ngmesh)


# --------------------------------- FE-SPACE ----------------------------------
X0 = L2(mesh, order=0)
X1 = L2(mesh, order=order)**5
X2 = L2(mesh, order=order)**3
X3 = L2(mesh, order=order)**8
X = L2(mesh, order=order)**14
IRS = IntegrationRuleSpace(mesh, order=order)**3

print(f'proc {rank}/{comm.size} has {X1.ndof}/{comm.Sum(X1.ndof)} dofs of X1')

gf_X = GridFunction(X)
gf_rho_d_per, gf_rho_w_per, gf_rhoU1, gf_rhoU2, gf_rhoE_per, \
    gf_rho_v_per, gf_rho_c_per, gf_T_per, gf_rho_d_0, gf_rho_w_0, \
    gf_rhoE_0, gf_rho_v_0, gf_rho_c_0, gf_T_0 = gf_X.components
gf_rhovel = CF((gf_rhoU1, gf_rhoU2))
gf_rho = gf_rho_d_0 + gf_rho_d_per + gf_rho_w_0 + gf_rho_w_per
gf_vel = gf_rhovel / gf_rho

gf_rhoe_per = gf_rhoE_0 + gf_rhoE_per - InnerProduct(gf_rhovel, gf_vel) / 2
gf_rhoe_per -= (c_vd * gf_rho_d_0 + c_vv * gf_rho_v_0 + c_l * gf_rho_c_0) * (gf_T_0 - T_ref)
gf_rhoe_per -= gf_rho_v_0 * (L_ref - R_v * T_ref)

gf_X3, gf_X_per = GridFunction(X3), GridFunction(X)
gf_X2, gfu_hydro = GridFunction(X2), GridFunction(X2)
gf_IRS = GridFunction(IRS)


# ----------------------------- (BI)LINEAR FORMS ------------------------------
U, V = X.TnT()

n = specialcf.normal(mesh.dim)


def Max(a, b):
    return IfPos(a - b, a, b)


def Min(a, b):
    return IfPos(a - b, b, a)


def Abs(u):
    return IfPos(u, u, -u)


# Flux function
def F(U):
    rho_d_per, rho_w_per, rhoU1, rhoU2, rhoE_per, rho_v_per, rho_c_per, T_per,\
        rho_d_0, rho_w_0, rhoE_0, rho_v_0, rho_c_0, T_0 = U

    rho_d = rho_d_0 + rho_d_per
    rho_w = rho_w_0 + rho_w_per

    rhovel = U[2:4]
    rho = rho_d + rho_w
    vel = rhovel / rho

    p_0 = R_d * rho_d_0 * T_0 + R_v * rho_v_0 * T_0

    p_per = R_d * rho_d_per * T_per + R_v * rho_v_per * T_per
    p_per += R_d * rho_d_per * T_0 + R_v * rho_v_per * T_0
    p_per += R_d * rho_d_0 * T_per + R_v * rho_v_0 * T_per

    F1 = rho_d * vel
    F2 = rho_w * vel
    F3 = OuterProduct(rhovel, vel) + p_per * Id(2)
    F4 = (rhoE_per + p_per + rhoE_0 + p_0) * vel
    Fz = CF((0, 0))

    return CF((F1, F2, F3, F4, Fz, Fz, Fz, Fz, Fz, Fz, Fz, Fz, Fz),
              dims=(14, 2))


# Speed of sound
def sound(U):
    rho_d_per, rho_w_per, rhoU1, rhoU2, rhoE_per, rho_v_per, rho_c_per, T_per,\
        rho_d_0, rho_w_0, rhoE_0, rho_v_0, rho_c_0, T_0 = U

    rho = rho_d_per + rho_d_0 + rho_w_per + rho_w_0
    p = (R_d * (rho_d_per + rho_d_0) + R_v * (rho_v_per + rho_v_0)) * (T_per + T_0)
    q_v = (rho_v_per + rho_v_0) / (rho_d_per + rho_d_0)
    q_c = (rho_c_per + rho_c_0) / (rho_d_per + rho_d_0)

    gamma = 1 + (R_d + R_v * q_v) / (c_vd + c_vv * q_v + c_l * q_c)

    return sqrt(gamma * p / rho)


# Maximal characteristic speed
def LamMax(A, B):
    vel_A = A[2:4] / (A[0] + A[1] + A[8] + A[9])
    vel_B = B[2:4] / (B[0] + B[1] + B[8] + B[9])
    lam_A = Abs(InnerProduct(vel_A, n)) + sound(A)
    lam_B = Abs(InnerProduct(vel_B, n)) + sound(B)
    return Max(lam_A, lam_B)


# Numerical flux (local Lax-Friedrich)
def Fhatn(U):
    Ubnd = CF((U[0:2],
               (1 - 2 * n[0]**2) * U[2] - 2 * n[0] * n[1] * U[3],
               -2 * n[0] * n[1] * U[2] + (1 - 2 * n[1]**2) * U[3],
               U[4:]))
    Uhat = U.Other(bnd=Ubnd)
    _Fhat = 1 / 2 * (F(U) + F(Uhat)) * n + 1 / 2 * LamMax(U, Uhat) * (U - Uhat)
    return _Fhat


def G(U):
    rho_d_per, rho_w_per, rhoU1, rhoU2, rhoE_per, rho_v_per, rho_c_per, T_per,\
        rho_d_0, rho_w_0, rhoE_0, rho_v_0, rho_c_0, T_0 = U
    rho_per = rho_d_per + rho_w_per
    G1 = 0
    G2 = 0
    G31 = rho_per * g
    G32 = 0
    G4 = rhoU1 * g
    return CF((G1, G2, G31, G32, G4, 0, 0, 0, 0, 0, 0, 0, 0, 0))


# -------------------------------- INTEGRATORS --------------------------------
comp = {'realcompile': compile_flag, 'wait': wait_compile}

dX = dx(bonus_intorder=order)
dS_int = dx(skeleton=True, bonus_intorder=order)
dS_bnd = ds(skeleton=True, bonus_intorder=order)

a = BilinearForm(X, nonassemble=True)
a += - InnerProduct(F(U), Grad(V)).Compile(**comp) * dX
a += InnerProduct(Fhatn(U)[:5], (V - V.Other())[:5]).Compile(**comp) * dS_int
a += InnerProduct(Fhatn(U)[:5], V[:5]).Compile(**comp) * dS_bnd
a += InnerProduct(G(U), V).Compile(**comp) * dX


# --------------------------- ARTIFICIAL DIFFUSION ----------------------------
def add_diffusion(a, U, V):
    def avg(u0, u1):
        return (u0 + u1) * 0.5

    h = specialcf.mesh_size
    n = specialcf.normal(mesh.dim)
    diff = scale_diff * magnitude * 0.5
    _alpha = alpha * order**2 / h

    Ubnd = CF((U[0:2],
               (1 - 2 * n[0]**2) * U[2] - 2 * n[0] * n[1] * U[3],
               -2 * n[0] * n[1] * U[2] + (1 - 2 * n[1]**2) * U[3],
               U[4:]))
    Uhat = U.Other(bnd=Ubnd)

    u, u_oth = U[:5], Uhat[:5]
    grad_u, grad_u_oth = Grad(U)[:5, :], Grad(Uhat)[:5, :]
    v, v_oth = V[:5], V.Other()[:5]
    grad_v, grad_v_oth = Grad(V)[:5, :], Grad(V.Other())[:5, :]

    diff_vol = diff * InnerProduct(grad_u, grad_v)

    diff_int = -diff * InnerProduct(avg(grad_u, grad_u_oth) * n, v - v_oth)
    diff_int += -diff * InnerProduct(avg(grad_v, grad_v_oth) * n, u - u_oth)
    diff_int += diff * _alpha * InnerProduct(u - u_oth, v - v_oth)

    diff_bnd = -diff * InnerProduct(avg(grad_u, grad_u_oth) * n, v)
    diff_bnd += -diff * InnerProduct(grad_v * n, u - u_oth)
    diff_bnd += diff * _alpha * InnerProduct(u - u_oth, v)

    a += diff_vol.Compile(**comp) * dX
    a += diff_int.Compile(**comp) * dS_int
    a += diff_bnd.Compile(**comp) * dS_bnd

    return None


gf_elwise_const, magnitude = GridFunction(X0), GridFunction(H1(mesh, order=1))
magnitude.vec.data[:] = 0

if artificial_diffusion:
    add_diffusion(a, U, V)


# --------------------------- TEMPERATURE PROBLEM ----------------------------
def clausius_clapeyron(T):
    p_vs = e_ref * (T / T_ref)**((c_pv - c_l) / R_v)
    p_vs *= exp(((L_ref - (c_pv - c_l) * T_ref) / R_v) * (1 / T_ref - 1 / T))
    return p_vs


def energy_problem(U_irs):
    rho_v_per, rho_c_per, T_per = U_irs

    rho_vs = clausius_clapeyron(T_per + gf_T_0) / (R_v * (T_per + gf_T_0))
    _rho_v_per = Min(rho_vs, gf_rho_w_per + gf_rho_w_0) - gf_rho_v_0
    _rho_c_per = gf_rho_w_per - _rho_v_per

    _rhoe_per = (c_vd * gf_rho_d_per + c_vv * _rho_v_per + c_l * _rho_c_per) * (gf_T_0 - T_ref)
    _rhoe_per += _rho_v_per * (L_ref - R_v * T_ref)
    _rhoe_per += (c_vd * gf_rho_d_0 + c_vv * gf_rho_v_0 + c_l * gf_rho_c_0) * T_per
    _rhoe_per += (c_vd * gf_rho_d_per + c_vv * _rho_v_per + c_l * _rho_c_per) * T_per

    f1 = _rho_v_per - rho_v_per
    f2 = _rho_c_per - rho_c_per
    f3 = _rhoe_per - gf_rhoe_per

    return CF((f1, f2, f3))


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
def theta_e_func(T, rho_v, rho_d):
    f = T * ((rho_d * R_d * T / p_ref)**(-R_d / (c_pd + c_l * q_t)))
    f *= exp((L_ref + (c_pv - c_l) * (T - T_ref)) * (rho_v / rho_d)
             / ((c_pd + c_l * q_t) * T))
    return f


mesh1d = Make1DMesh(n=int(ceil(domain_h / h_max)),
                    mapping=lambda x: x * domain_h)

X1_1d = H1(mesh1d, order=order + 1, dirichlet='left')
X2_1d = L2(mesh1d, order=order)**3
X_1d = X1_1d * X2_1d

gfu_1d, gfu_1d_2 = GridFunction(X_1d), GridFunction(X_1d)
gfu_1d.vec[:], gfu_1d_2.vec[:] = 0.0, 0.0


def initial_constraint_residual(p, U):
    T, rho_vs, rho_d = U

    res1 = p - T * (rho_d * R_d + rho_vs * R_v)
    res2 = rho_vs * R_v * T - clausius_clapeyron(T)
    res3 = theta_e_barx - theta_e_func(T, rho_vs, rho_d)

    return CF((res1, res2, res3))


if rank == 0 and restart_flag is False:
    (p_1d, U_1d), (q_1d, V_1d) = X_1d.TnT()

    a_1d = BilinearForm(X_1d)
    a_1d += (Grad(p_1d) + (1 + q_t) * g * U_1d[2]) * q_1d * dx
    a_1d += InnerProduct(initial_constraint_residual(p_1d, U_1d), V_1d) * dx

    gfu_1d.components[0].Set(p_ref)
    gfu_1d_2.components[1].Set((T_ref, 0.01, 1.2))
    gfu_1d.vec.data += gfu_1d_2.vec

    Newton(a_1d, gfu_1d, maxerr=1e-10)
    del a_1d, p_1d, U_1d, q_1d, V_1d

if comm.size > 1:
    comm.mpi4py.Bcast(gfu_1d.vec.FV().NumPy(), root=0)

if restart_flag is False:
    gfu_hydro.Set(gfu_1d.components[1])
    gf_T_h, gf_rho_vs_h, gf_rho_d_h = gfu_hydro

    gf_X.Set((0, 0, 0, 0, 0, 0, 0, 0,
              gf_rho_d_h,
              q_t * gf_rho_d_h,
              (gf_rho_d_h * c_vd + c_vv * gf_rho_vs_h
               + c_l * (gf_rho_d_h * q_t - gf_rho_vs_h)) * (gf_T_h - T_ref)
              + gf_rho_vs_h * (L_ref - R_v * T_ref),
              gf_rho_vs_h,
              gf_rho_d_h * q_t - gf_rho_vs_h,
              gf_T_h))

del mesh1d, X1_1d, X2_1d, gfu_1d, gfu_1d_2


# ----------------------------- INITIAL CONDITION -----------------------------
if restart_flag is False:
    theta_e_per = DeltaTheta / (1 + _a**-2 * (y - domain_w / 2)**2)
    theta_e_per *= sin(pi * x / domain_h)

    def initial_temp(U):
        T, rho_vs, rho_d = U

        _gf_pre = (R_d * gf_rho_d_0 + R_v * gf_rho_v_0) * gf_T_0
        _gf_theta_e = theta_e_barx + theta_e_per

        r1 = _gf_theta_e - theta_e_func(T, rho_vs, rho_d)
        r2 = rho_vs * R_v * T - clausius_clapeyron(T)
        r3 = _gf_pre - (rho_d * R_d + rho_vs * R_v) * T

        return CF((r1, r2, r3))

    initial_temp_problem = NewtonCF(initial_temp(u_irs).Compile(),
                                    gfu_hydro, tol=newton_tol,
                                    maxiter=newton_maxit)

    f_irs_initT = LinearForm(IRS)
    f_irs_initT += (initial_temp_problem * v_irs).Compile() * dx_irs

    mass_irs = BilinearForm(IRS, symmetric=True, diagonal=True)
    mass_irs += (u_irs * v_irs).Compile() * dx_irs

    mass_irs.Assemble()
    f_irs_initT.Assemble()
    gf_IRS.vec.data = mass_irs.mat.Inverse() * f_irs_initT.vec

    _init_T, _init_rho_vs, _init_rho_d = gf_IRS

    _init_rho_d_per = _init_rho_d - gf_rho_d_0
    _init_rho_w_per = (_init_rho_d - gf_rho_d_0) * q_t
    _init_rho_v_per = _init_rho_vs - gf_rho_v_0
    _init_rho_c_per = _init_rho_w_per - _init_rho_v_per
    _init_rhovelx = (1 + q_t) * _init_rho_d * vel_x_init

    _init_rhoE = c_vd * _init_rho_d + c_vv * _init_rho_vs
    _init_rhoE += c_l * (_init_rho_d * q_t - _init_rho_vs)
    _init_rhoE *= (_init_T - T_ref)
    _init_rhoE += _init_rho_vs * (L_ref - R_v * T_ref)
    _init_rhoE += 1 / 2 * (1 + q_t) * _init_rho_d * (0**2 + vel_x_init**2)

    _init_rhoe_0 = c_vd * gf_rho_d_0 + c_vv * gf_rho_v_0
    _init_rhoe_0 += c_l * gf_rho_c_0
    _init_rhoe_0 *= (gf_T_0 - T_ref)
    _init_rhoe_0 += gf_rho_v_0 * (L_ref - R_v * T_ref)
    _init_rhoE_per = _init_rhoE - _init_rhoe_0

    _init_T_per = _init_T - gf_T_0

    _init = CF((_init_rho_d_per, _init_rho_w_per, 0, _init_rhovelx,
                _init_rhoE_per, _init_rho_v_per, _init_rho_c_per, _init_T_per))

    f_init = LinearForm(X3)
    f_init += InnerProduct(_init, X3.TestFunction()).Compile() * dx_irs

    del _init_rho_d_per, _init_rho_w_per, _init_rho_v_per, _init_rho_c_per, \
        _init_rhoE, _init_rhoe_0, _init_rhoE_per, _init_T_per, _init, \
        mass_irs, f_irs_initT
else:
    raise Exception('Restarting not implemented yet')


# ------------------------------- VISUALISATION -------------------------------
gf_theta_e_per = theta_e_func(gf_T_0 + gf_T_per, gf_rho_v_0 + gf_rho_v_per,
                              gf_rho_d_0 + gf_rho_d_per) - theta_e_barx

if comm.size == 1:
    Draw(gf_rho_d_per, mesh, 'rho_d_per')
    Draw(gf_rho_w_per, mesh, 'rho_w_per')
    Draw(gf_rho_v_per, mesh, 'rho_v_per')
    Draw(gf_rho_c_per, mesh, 'rho_c_per')
    Draw(gf_vel, mesh, 'vel')
    Draw(gf_rhoE_per, mesh, 'rhoE_per')
    Draw(gf_T_per, mesh, 'T_per')
    Draw(gf_theta_e_per, mesh, 'theta_e_per')


# ---------------------------------- OUTPUT -----------------------------------
try:
    data_dir = os.environ['DATA']
except KeyError:
    print(f'rank {rank}: DATA environment variable does not exist')
    data_dir = '..'
if (vtk_flag or pickle_flag) and rank == 0:
    print(f'data will be saved in the directory {data_dir}')

comp_dir_name = os.getcwd().split('/')[-1]
pickle_dir_abs = f'{data_dir}/{comp_dir_name}/{pickle_dir}'

if pickle_flag:
    if rank == 0 and not os.path.isdir(pickle_dir_abs):
        os.makedirs(pickle_dir_abs)
    comm.Barrier()

    def do_pickle(it):
        filename = f'{pickle_dir_abs}/{pickle_name}_step{it}_rank{rank}.dat'
        pickler = pickle.Pickler(open(filename, 'wb'))
        pickler.dump(gf_X.vec)
        return None

if vtk_flag:
    vtk_dir_abs = f'{data_dir}/{comp_dir_name}/{vtk_dir}'
    if rank == 0 and not os.path.isdir(vtk_dir_abs):
        os.makedirs(vtk_dir_abs)
    comm.Barrier()

    vtk = VTKOutput(ma=mesh,
                    coefs=[gf_rho_d_0 + gf_rho_d_per,
                           gf_rho_v_0 + gf_rho_v_per,
                           gf_rho_c_0 + gf_rho_c_per,
                           gf_vel, gf_rhoE_0 + gf_rhoE_per, gf_T_per,
                           gf_theta_e_per, magnitude],
                    names=['rho_d', 'rho_v', 'rho_c', 'vel', 'rhoE', 'T_per',
                           'theta_e_pert', 'magnitude'],
                    filename=vtk_dir_abs + '/' + vtk_name,
                    subdivision=vtk_subdiv,
                    order=vtk_order,
                    floatsize='single',
                    legacy=False)


# ------------------------------- TIME STEPPING -------------------------------
if comm.size == 1:
    tm = TaskManager()
    tm.__enter__()

res, U0, U1, U2, U3 = [gf_X.vec.CreateVector() for i in range(5)]

if pickle_flag:
    do_pickle(-1)

f_init.Assemble()
gf_X3.vec.data = X3.Mass(1).Inverse() * f_init.vec
del f_init

gf_X_per.vec.data[:] = 0
gf_X_per.vec.data[:X3.ndof] = gf_X3.vec
gf_X.vec.data += gf_X_per.vec

inv_m = X1.Mass(1).Inverse()

if pickle_flag:
    do_pickle(0)
if vtk_flag:
    vtk.Do(time=0.0)

for it in range(1, int(t_end / dt) + 1):
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
        norm_vel = Integrate(InnerProduct(gf_vel, gf_vel), mesh,
                             order=2 * order, element_wise=True)
        gf_elwise_const.vec.FV().NumPy()[:] = norm_vel
        magnitude.Set(sqrt(gf_elwise_const))

    if comm.size == 0:
        Redraw(blocking=False)
    if pickle_flag and it % pickle_freq == 0:
        do_pickle(it)
    if vtk_flag and it % vtk_freq == 0:
        vtk.Do(time=dt * it)
    if rank == 1 or comm.size == 1:
        print(f't={dt * it:11.7f}', end='\r')

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
