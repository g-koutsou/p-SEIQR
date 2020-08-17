from matplotlib import pyplot as plt
import matplotlib as mpl
import collections
import numpy as np
import gvar as gv
import datetime
import json

cmd = "./p-SEIQR/c-SIR/c-sir"
trajs = "./data/trajs/traj-tail-n00250000.txt"
cy_data = np.loadtxt("./data/cyprus-confirmed.txt")
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

def R0(vel, prob=1):
    if prob == 1:
        p = gv.gvar(["3.639(54)", "-0.0885(91)"])
        return p[0]*(vel + p[1])
    else:
        p = gv.gvar(["3.675(35)", "0.011(15)", "-0.366(21)", "-0.0120(83)"])
        return p[0]*vel*prob + p[1]*vel + p[2]*prob + p[3]

def R0_int(c):
    if len(c.shape) == 1:
        c = c.reshape([1,c.shape[0]])
    t_h = t_e+t_i
    rho = c[:,1:] - c[:,:-1]
    t = rho.shape[1]
    den = np.array([rho[:,i-t_h+1:i-t_e].sum(axis=1) for i in range(t_h-1, t-1)]).T
    num = rho[:,t_h-1:t][:,1:]
    x = np.arange(t_h-1, t-1)
    if np.any(den == 0):
        i = np.min(np.where(den == 0)[1])
        num = num[:,:i]
        den = den[:,:i]
        x = x[:i]
    be = (num/den)
    return x,be.squeeze()*(t_i)

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

N_days = 297

day0 = datetime.datetime(2020,3,9)

scenarios = json.load(open("p-SEIQR/analysis/scenarios.json", "r"))

scens = ("A","B","C","D")

xs = collections.defaultdict(list)
ys = collections.defaultdict(list)
for scen in scens:
    for s in range(32):
        x,y = np.loadtxt("data/scenarios/scen-{}-seed{:06.0f}.txt".format(scen, s)).T
        xs[scen].append(x)
        ys[scen].append(y)

descr = ["Continue with fitted $R^{model}_0(t)$\n to end of year",
         "Increase detection rate\nto $\\sim50\%$",
         "Reduce $R^{model}_0(t)$ to $\\sim$1\non August 1$^{st}$",
         "Impose strict lockdown on\n20% of population"]

color_wheel = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig = plt.figure(1, figsize=(1+4*len(scens)//2,11.5))
fig.clf()
gs = mpl.gridspec.GridSpec(2*14, len(scens)//2, wspace=0.075, left=0.075, right=0.95, bottom=0.12, top=0.95)
for i,scen in enumerate(scens):
    col = i%2
    row = i//2
    ax = fig.add_subplot(gs[row*14:row*14+8, col])
    x = np.linspace(0, N_days/ascale, N_days)
    y = np.array([np.interp(x, xp, yp) for (xp,yp) in zip(xs[scen],ys[scen])])
    av = y.mean(axis=0)
    lo = np.percentile(y, 10, axis=0)
    hi = np.percentile(y, 90, axis=0)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,_: "{:4.1f}".format(x/1000)))
    m0, = ax.plot(cy_data, color="k", marker="o", ls="", ms=4, alpha=0.6)
    m1, = ax.plot(x*ascale, av, ls="-")    
    ax.fill_between(x*ascale, lo, hi, color=color_wheel[0], alpha=0.1)
    ax.set_xlim(0, N_days)
    ax.set_xticklabels([])
    ax.set_ylim(0, 1_330)
    if i == 0:
        ax.legend([m0,m1], ["Cyprus data", "Particle model"], loc="lower right", frameon=False)
    if col == 0:
        ax.set_ylabel(r"Confirmed cases ($\times 10^3$)")
    else:
        ax.set_yticklabels([])    
    ax.text(N_days/2, 800, "Scenario {}\n{}".format("ABCD"[i], descr[i]),
            ha="center", va="top", fontsize=11)
for i,scen in enumerate(scens):
    col = i%2
    row = i//2
    ax = fig.add_subplot(gs[row*14+8:row*14+11, col])
    x = np.linspace(0, N_days/ascale, N_days)
    y = np.array([np.interp(x, xp, yp) for (xp,yp) in zip(xs[scen],ys[scen])])
    av = y.mean(axis=0)
    vel_params = scenarios[scen]["vel_params"]
    pro_params = scenarios[scen]["pro_params"]
    tv_params = scenarios[scen]["tv_params"]
    ts = np.arange(0, N_days, 1)
    vels = sigmoids(vel_params, tv_params)(ts/ascale)
    vv = np.array([vels[:-1],vels[:-1]]).T.flatten()
    tt = np.array([ts[:-1],ts[1:]]).T.flatten()
    if np.all(np.array([len(p['pop']) for p in pro_params]) == 1):
        a,e = gv.mean(R0(vv)),gv.sdev(R0(vv))
        ax.plot(tt, a, ls="-", color=color_wheel[0], label="R$^{model}_0$")
        ax.fill_between(tt, a+e, a-e, alpha=0.5, color=color_wheel[0])
    else:
        for j,p in enumerate(pro_params):
            t0 = pro_params[j]["t"]
            t1 = pro_params[j+1]["t"] if j+1 != len(pro_params) else np.inf
            v = vv[np.logical_and(tt>=t0*ascale, tt<=t1*ascale)]
            t = tt[np.logical_and(tt>=t0*ascale, tt<=t1*ascale)]
            for k,(pop,prb) in enumerate(zip(p["pop"],p["prb"])):
                a,e = gv.mean(R0(v, prob=prb)),gv.sdev(R0(v, prob=prb))
                ax.plot(t, a, ls=["-","-."][k], color=color_wheel[0])
                ax.fill_between(t, a+e, a-e, color=color_wheel[0], alpha=0.2)
    t,r0 = R0_int(av)
    ax.plot(t, r0, ls="--", color=color_wheel[1], label="R$^{integral}_0$")
    # ax.plot(np.arange(t_i+t_e+1, cy_data.shape[0]), R0_int(cy_data), ls="-", color=color_wheel[2])
    ax.set_ylim(-0.1, 2.7)
    ax.set_xlim(0, N_days)
    ax.set_xticklabels([])
    if i == 0:
        ax.legend(loc="upper right", frameon=False, ncol=2)
    if col == 0:
        ax.set_ylabel("R$_0$")
    else:
        ax.set_yticklabels([])        
for i,scen in enumerate(scens):
    col = i%2
    row = i//2
    qra_params = scenarios[scen]["qra_params"]
    tr_params = scenarios[scen]["tr_params"]
    ax = fig.add_subplot(gs[row*14+11:row*14+13, col])
    ts = np.arange(0, N_days, 1)
    rat = sigmoids(qra_params, tr_params)(ts/ascale)
    rat = np.array([rat[:-1],rat[:-1]]).T.flatten()
    tt = np.array([ts[:-1],ts[1:]]).T.flatten()
    ax.plot(tt, 1-gv.mean(rat), ls="-", color="k")
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlim(0, N_days)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,_: "{2:02.0f}/{1:02.0f}\n{0}".format(
        *((day0 + datetime.timedelta(x)).timetuple()[:3]))))
    if col == 0:
        ax.set_ylabel("r(t)")
    else:
        ax.set_yticklabels([])
    if row == 0:
        ax.set_xticklabels([])
fig.canvas.draw()
fig.show()

if False:
    fig.patch.set_facecolor(None)
    fig.patch.set_alpha(0)
    fig.savefig("particle-scenarios.pdf", bbox_inches='tight', facecolor=fig.get_facecolor())
