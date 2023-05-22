# ------------------------------ LOAD LIBRARIES -------------------------------
from ngsolve import *
from ngsolve.meshes import MakeStructured2DMesh
import os
import pickle

SetNumThreads(128)

# -------------------------------- PARAMETERS ---------------------------------
levels = [{'k': 1, 'h': 1000., 'dt': 1.0},
          {'k': 1, 'h': 500.0, 'dt': 0.5},
          {'k': 1, 'h': 250.0, 'dt': 0.25},
          {'k': 1, 'h': 125.0, 'dt': 0.125},
          #
          {'k': 2, 'h': 1000., 'dt': 0.5},
          {'k': 2, 'h': 500.0, 'dt': 0.25},
          {'k': 2, 'h': 250.0, 'dt': 0.125},
          {'k': 2, 'h': 125.0, 'dt': 0.0625},
          #
          {'k': 3, 'h': 1000., 'dt': 0.3},
          {'k': 3, 'h': 500.0, 'dt': 0.15},
          {'k': 3, 'h': 250.0, 'dt': 0.075},
          {'k': 3, 'h': 125.0, 'dt': 0.0375},
          #
          {'k': 4, 'h': 1000., 'dt': 0.2},
          {'k': 4, 'h': 500.0, 'dt': 0.1},
          {'k': 4, 'h': 250.0, 'dt': 0.05},
          ]
orders = list(set(([_d['k'] for _d in levels])))

n_files = 30
base_name = 'cloudy_gravity_waves_dg_ssprk43_artdiff0quads1struc1_'

if 'DATA' in os.environ:
    data_dir = os.environ['DATA']
else:
    data_dir = '..'
comp_dir_name = os.getcwd().split('/')[-1]
pickle_dir = 'pickle_output'
pickle_dir_abs = f'{data_dir}/{comp_dir_name}/{pickle_dir}'

order_fine = 4
h_fine = 125.0
dt_fine = 0.025

dump_file = f'results_{base_name}raw.dat'

# ----------------------------------- DATA ------------------------------------
c_l = 4186                  # Specific heat of water
c_pd = 1004                 # Specific heat of dry air at constant pressure
c_pv = 1885                 # Specific heat of vapour at constant pressure
c_vd = 717                  # Specific heat of dry air at constant volume
c_vv = 1424                 # Specific heat of vapour at constant volume
R_d = c_pd - c_vd           # Gas constant for dry air
R_v = c_pv - c_vv           # Gas constant for water vapour
L_ref = 2.5e6               # Latent heat of vaporization at T_ref
T_ref = 273.15              # Reference temperature
p_ref = 1e5                 # Reference pressure
g = 9.81                    # Acceleration due to gravity

q_t = 0.02                  # Total water mixing ratio
Theta_0 = 300
N2 = 1e-4
theta_e_bar = Theta_0 * exp(N2 * x / g)  # Do everything with swapped x/y

t_end = 3600
dom_h = 10000
dom_w = 300000


# --------------------------------- FUNCTIONS ---------------------------------
def theta_e_func(T, rho_vs, rho_d):
    f = T * ((rho_d * R_d * T / p_ref)**(-R_d / (c_pd + c_l * q_t)))
    f *= exp((L_ref + (c_pv - c_l) * (T - T_ref)) * (rho_vs / rho_d)
             / ((c_pd + c_l * q_t) * T))
    return f


def l2_norm(cf):
    return sqrt(Integrate(InnerProduct(cf, cf).Compile(), mesh_fine,
                          order=2 * order_fine))


# -------------------------- PREPARE REFERENCE DATA ---------------------------
mesh_fine = MakeStructured2DMesh(quads=True,
                                 nx=int(ceil(dom_h / h_fine)),
                                 ny=int(ceil(dom_w / h_fine)),
                                 periodic_y=True,
                                 mapping=lambda x, y: (x * dom_h, y * dom_w))

X2 = L2(mesh_fine, order=order_fine)**14
gf_X2 = GridFunction(X2)

rho_d_per_2, rho_w_per_2, rhoU12, rhoU22, rhoE_per_2, rho_v_per_2, \
    rho_c_per_2, T_per_2, rho_d_0_2, rho_w_0_2, rhoE_0_2, rho_v_0_2, \
    rho_c_0_2, T_0_2 = gf_X2.components

rho_d2 = rho_d_0_2 + rho_d_per_2
rho_w2 = rho_w_0_2 + rho_w_per_2
rho_v2 = rho_v_0_2 + rho_v_per_2
rho_c2 = rho_c_0_2 + rho_c_per_2
rhovel2 = CF((rhoU12, rhoU22))
rho2 = rho_d2 + rho_w2
vel2 = rhovel2 / rho2
rhoE2 = rhoE_0_2 + rhoE_per_2
T2 = T_0_2 + T_per_2
theta_e2 = theta_e_func(T2, rho_v2, rho_d2)

results = {_k: [] for _k in orders}

try:
    unpickler = pickle.Unpickler(open(dump_file, 'rb'))
    dump = unpickler.load()
    del unpickler
    print('Loaded data', dump)
except FileNotFoundError:
    dump = {}

# ----------------------------- MESH/ORDER LOOP -------------------------------
for par in levels:
    print('pars = ', par)

    dump_key = f'{par["k"]}_{int(par["h"])}_{par["dt"]:6.4f}'

    try:
        errs = dump[dump_key]
    except KeyError:
        mesh = MakeStructured2DMesh(
            quads=True, nx=int(ceil(dom_h / par["h"])),
            ny=int(ceil(dom_w / par["h"])), periodic_y=True,
            mapping=lambda x, y: (x * dom_h, y * dom_w))

        X = L2(mesh, order=par["k"])**14
        gf_X = GridFunction(X)
        rho_d_per, rho_w_per, rhoU1, rhoU2, rhoE_per, rho_v_per, rho_c_per, \
            T_per, rho_d_0, rho_w_0, rhoE_0, rho_v_0, rho_c_0, T_0 \
            = gf_X.components

        rho_d = rho_d_0 + rho_d_per
        rho_w = rho_w_0 + rho_w_per
        rho_v = rho_v_0 + rho_v_per
        rho_c = rho_c_0 + rho_c_per
        rhovel = CF((rhoU1, rhoU2))
        rho = rho_d + rho_w
        vel = rhovel / rho
        rhoE = rhoE_0 + rhoE_per
        T = T_0 + T_per
        theta_e = theta_e_func(T, rho_v, rho_d)

        errs = {'rho_d': [], 'rho_w': [], 'rho_v': [], 'rho_c': [], 'rhoV': [],
                'velX': [], 'velY': [], 'rhoE': [], 'T': [], 'theta_e': []}

        pickle_name = base_name + f'h{par["h"]}k{par["k"]}dt{par["dt"]}'
        pickle_freq = int(t_end / par['dt'] / n_files)

        pickle_name2 = base_name + f'h{h_fine}k{order_fine}dt{dt_fine}'
        pickle_freq2 = int(t_end / dt_fine / n_files)

        with TaskManager():
            for i in range(0, int(n_files + 1)):
                print(f'Computing errors: Step {i} / {n_files + 1}', end='\r')
                it = i * pickle_freq
                in_file = f'{pickle_dir_abs}/{pickle_name}_step{it}_rank0.dat'
                unpickler = pickle.Unpickler(open(in_file, 'rb'))
                gf_X.vec.data = unpickler.load()

                it = i * pickle_freq2
                in_file = f'{pickle_dir_abs}/{pickle_name2}_step{it}_rank0.dat'
                unpickler = pickle.Unpickler(open(in_file, 'rb'))
                gf_X2.vec.data = unpickler.load()

                errs['rho_d'].append(l2_norm(rho_d - rho_d2))
                errs['rho_w'].append(l2_norm(rho_w - rho_w2))
                errs['rho_v'].append(l2_norm(rho_v - rho_v2))
                errs['rho_c'].append(l2_norm(rho_c - rho_c2))
                errs['rhoV'].append(l2_norm(rhovel - rhovel2))
                errs['velX'].append(l2_norm(vel[0] - vel2[0]))
                errs['velY'].append(l2_norm(vel[1] - vel2[1]))
                errs['rhoE'].append(l2_norm(rhoE - rhoE2))
                errs['T'].append(l2_norm(T - T2))
                errs['theta_e'].append(l2_norm(theta_e - theta_e2))

        dump[dump_key] = errs
        pickler = pickle.Pickler(open(dump_file, 'wb'))
        pickler.dump(dump)
        del pickler

    l2l2 = {}
    for key, err in errs.items():
        l2l2[key] = sqrt(sum([e**2 for e in err]))
    results[par['k']].append(l2l2)
    print('')

# ----------------------- WRITE TO HUMAN READABLE FILE ------------------------
for order in orders:
    file_out = 'results_' + base_name + f'order{order}_pert.txt'
    with open(file_out, 'w') as fid:
        fid.write('lvl')
        keys = list(l2l2.keys())
        for key in keys:
            fid.write(f' {key}')
        fid.write('\n')
        for i, errs in enumerate(results[order]):
            fid.write(f'{i}')
            for key in keys:
                fid.write(f' {errs[key]:6.4e}')
            fid.write('\n')
