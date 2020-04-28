import tqdm
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from collections import defaultdict
import gvar as gv
import lsqfit

rt = 4.38
vs = np.arange(0.1, 0.675, 0.025)
seeds = list(range(1, 10))
ascale = 10/rt # days per time unit

sets = dict()
for v in vs:
    sets[v] = defaultdict(list)
    for s in seeds:
        fname = "data/v-scan/out-s{:02.0f}-t{:4.2f}-v{:5.3f}.txt".format(s,rt,v)
        for line in tqdm.tqdm(open(fname, "r"), desc="v={:5.3f}, seed={:2.0f}".format(v, s)):
            i,t,Rt,S,I,Q,RI,RQ = line.split()
            sets[v][s,int(i)].append((float(t), float(Rt), int(S), int(I), int(Q), int(RI), int(RQ)))
    sets[v] = {k: np.array(sets[v][k]) for k in sets[v]}

def interpolate(xs, ys, xp=None):
    if xp is None:
        maxx = np.concatenate(xs).max()
        minx = np.concatenate(xs).min()
        xp = np.linspace(minx, maxx, 75)
    yp = np.array([np.interp(xp, x, y) for x,y in zip(xs,ys)])
    return xp,yp

fig = plt.figure(1)
fig.clf()
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
for v in vs:
    t = [sets[v][x][:,0] for x in sets[v]]
    I = [sets[v][x][:,3] for x in sets[v]]
    tp,yp = interpolate(t, I)
    nstat = yp.shape[0]
    ave = yp.mean(axis=0)
    err = yp.std(axis=0)/np.sqrt(yp.shape[0])
    m, = ax.plot(tp*ascale, ave, ls="-", lw=0.5, label=r"$v$={:5.3f}, $N_{{s}}$={}".format(v, nstat))
    ax.fill_between(tp*ascale, ave+err, ave-err, alpha=0.5, color=m.get_color())
ax.legend(loc="upper left", frameon=False, ncol=2)
ax.set_ylabel("I(t)")
ax.set_xlabel("t [days]")
fig.canvas.draw()
fig.show()

fig = plt.figure(2)
fig.clf()
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
for v in vs:
    t = [sets[v][x][:,0] for x in sets[v]]
    R = [sets[v][x][:,1] for x in sets[v]]
    tp,yp = interpolate(t, R)
    nstat = yp.shape[0]
    ave = yp.mean(axis=0)
    err = yp.std(axis=0)/np.sqrt(yp.shape[0])
    m, = ax.plot(tp*ascale, ave, ls="-", lw=0.5, label=r"$v$={:5.3f}, $N_{{s}}$={}".format(v, nstat))
    ax.fill_between(tp*ascale, ave+err, ave-err, alpha=0.5, color=m.get_color())
    ax.plot([rt*ascale, rt*ascale], [0, max(ave)], ls="--", color=m.get_color())
ax.legend(loc="upper left", frameon=False, ncol=3)
ax.set_ylim(0, 4)
ax.set_xlim(0,28)
ax.set_ylabel("R(t)")
ax.set_xlabel("t [days]")
fig.canvas.draw()
fig.show()

fig = plt.figure(3)
fig.clf()
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
data = list()
for v in vs:
    t = [sets[v][x][:,0] for x in sets[v]]
    R = [sets[v][x][:,1] for x in sets[v]]
    ### Returns times between rt/2 and 3*rt/2
    tsl = [x[np.logical_and(rt/2 < x, x < 3*rt/2)] for x in t]
    ### Get the average step between consecutive times
    dt = np.array([(x[1:]-x[:-1]).mean() for x in tsl if len(x) > 1])
    dt = dt[np.isnan(dt) == False].mean()
    tp,yp = interpolate(t, R, xp=rt+dt)
    ave = yp.mean(axis=0)
    err = yp.std(axis=0)/np.sqrt(yp.shape[0])
    data.append((v, ave, err))
x,ave,err = np.array(list(zip(*data)))
m,_,_ = ax.errorbar(x, ave, err, ls="", marker="o")
fit = lsqfit.nonlinear_fit(data=(x, gv.gvar(ave, err)), fcn=lambda x,p: p[0]*(x+p[1]), p0=[1,1])
x = np.linspace(min(x)-0.01, max(x)+0.01)
y = fit.p[0]*(x+fit.p[1])
ax.plot(x, gv.mean(y), ls="-", color=m.get_color())
ax.fill_between(x, gv.mean(y)+gv.sdev(y), gv.mean(y)-gv.sdev(y), color=m.get_color(), alpha=0.2)
x = np.linspace(0, max(x)+1)
y = fit.p[0]*(x+fit.p[1])
ax.plot(x, gv.mean(y), ls=":", color=m.get_color())
ax.set_xlim(0, 0.721)
ax.set_ylim(0, 3.1)
ax.set_ylabel(r"$R_0$=R($\tau$)")
ax.set_xlabel(r"$v$")
fig.canvas.draw()
fig.show()

if False:
    for i in [1,2,3]:
        plt.figure(i).savefig("v-scan-{}.pdf".format(i), bbox_inches="tight") 
