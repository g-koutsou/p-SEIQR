from matplotlib import pyplot as plt
import matplotlib as mpl
import multiprocess
import scipy.optimize
import numpy as np
import scipy as sp
import subprocess
import gvar as gv
import tempfile
import datetime
import json

cmd = "./p-SEIQR/c-SIR/c-sir"
trajs = "./data/trajs/traj-tail-n00250000.txt"
cy_data = np.loadtxt("cy.txt")
T0,T1,T2 = 15,73,145
Ti = 0

t_i = 10
t_e = 2
tau_i = 3.617
tau_e = tau_i*0.2
ascale = t_i/tau_i
ti,t0,t1,t2 = Ti/ascale,T0/ascale,T1/ascale,T2/ascale

def vel(R0):
    p = gv.gvar(["3.639(54)", "-0.0885(91)"])
    return R0/p[0] - p[1]

def R0(vel):
    p = gv.gvar(["3.639(54)", "-0.0885(91)"])
    return p[0]*(vel + p[1])

def R0_int(c):
    t_h = t_e+t_i
    rho = c[1:] - c[:-1]
    t = len(rho)
    be = np.array([rho[i+1]/rho[i-t_h+1:i-t_e].sum() for i in range(t_h-1,t-1)])
    return be*(t_i)

def sigmoids(p, T):
    assert (len(p) - 1) % 2 == 0
    ns = (len(p)-1)//2
    assert len(T) == ns
    heights = p[:(ns+1)]
    masses = p[(ns+1):]
    def s(x):
        acc = 0
        for i in range(ns):
            m = masses[i]
            d = heights[i+1]-heights[i]
            acc += d*np.tanh(m*(x - T[i]))
        acc += heights[-1] + heights[0]
        return acc/2
    return s

def func(p, seed, N_days):
    T = N_days/ascale
    t = ti + np.arange(0, T, 1/ascale) 
    vels = sigmoids(p, tv_params)(t)
    quar = sigmoids(qra_params, tr_params)(t)
    with tempfile.NamedTemporaryFile() as fv:
        np.savetxt(fv, np.array([t-ti, vels]).T)
        fv.file.flush()
        with tempfile.NamedTemporaryFile() as fq:
            np.savetxt(fq, np.array([t-ti, quar]).T)
            fq.file.flush()
            with tempfile.NamedTemporaryFile() as fp:
                for pp in pro_params:
                    t,pop,prb = pp["t"],pp["pop"],pp["prb"]
                    s = ["{}:{}".format(x,y) for x,y in zip(prb,pop)]
                    fp.write(str.encode("{} {}\n".format(t, " ".join(s))))
                fp.file.flush()
                c = list()
                c.append(r"{}".format(cmd))
                c.append(r"-s {}".format(seed))
                c.append(r"-I {}".format(I0))
                c.append(r"-E {}".format(E0))
                c.append(r"-Q {}".format(Q0))
                c.append(r"-t {}".format(tau_i))
                c.append(r"-e {}".format(tau_e))
                c.append(r"-T {}".format(T))
                c.append(r"--probability-profile={}".format(fp.name))
                c.append(r"--velocity-profile={}".format(fv.name))
                c.append(r"--quarantine-profile={}".format(fq.name))
                c.append(r"{}".format(Npop))
                c.append(r"{}".format(trajs))
                r = subprocess.run(c, stdout=subprocess.PIPE)
    d = np.array(list(map(str.split, r.stdout.decode().split("\n")[:-1])), float)
    return d[:,1]+ti,d[:,6]+d[:,8]

N_days = 297

day0 = datetime.datetime(2020,3,9)

scenarios = json.load(open("p-SEIQR/analysis/scenarios.json", "r"))

scen = "C"

vel_params = scenarios[scen]["vel_params"]
tv_params = scenarios[scen]["tv_params"]
qra_params = scenarios[scen]["qra_params"]
tr_params = scenarios[scen]["tr_params"]
pro_params = scenarios[scen]["pro_params"]

Npop = 250_000
rQ = qra_params[0]
Q0 = int(cy_data[Ti])
Q1 = int(cy_data[Ti+1])
Q2 = int(cy_data[Ti+2])
I0 = int(Q0*(1-rQ)/rQ)
E0 = 100

prms = list()
for s in range(32):
    prms.append((vel_params, s, N_days))

with multiprocess.Pool(None) as pool:
    r = pool.starmap_async(func, prms)
    r.wait()
    xs,ys = zip(*[x for x in r.get()])

for s,(x,y) in enumerate(zip(xs,ys)):
    np.savetxt("data/scenarios/scen-{}-seed{:06.0f}.txt".format(scen, s), np.array([x,y]).T)
