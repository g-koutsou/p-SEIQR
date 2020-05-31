import tqdm
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from collections import defaultdict
import gvar as gv
import lsqfit

rts = np.arange(0.5, 5.5, 0.5)
seeds = [7, 8, 9]

sets = dict()
for rt in rts:
    sets[rt] = defaultdict(list)
    for s in seeds:
        fname = "data/rt-scan/out-s{:02.0f}-t{:4.2f}.txt".format(s,rt)
        for line in tqdm.tqdm(open(fname, "r"), desc="t={:4.2f}, seed={:d}".format(rt, s)):
            i,t,Rt,S,I,Q,RI,RQ = line.split()
            sets[rt][s,int(i)].append((float(t), float(Rt), int(S), int(I), int(Q), int(RI), int(RQ)))
    sets[rt] = {k: np.array(sets[rt][k]) for k in sets[rt]}

def interpolate(xs, ys, xp=None):
    if xp is None:
        maxx = np.concatenate(xs).max()
        minx = np.concatenate(xs).min()
        xp = np.linspace(minx, maxx, 1024)
    yp = np.array([np.interp(xp, x, y) for x,y in zip(xs,ys)])
    return xp,yp

fig = plt.figure(1)
fig.clf()
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
for rt in rts:
    t = [sets[rt][x][:,0] for x in sets[rt]]
    I = [sets[rt][x][:,3] for x in sets[rt]]
    tp,yp = interpolate(t, I)
    nstat = yp.shape[0]
    ave = yp.mean(axis=0)
    err = yp.std(axis=0)/np.sqrt(yp.shape[0])
    m, = ax.plot(tp, ave, ls="-", lw=0.5, label=r"$\tau$={:4.2f}, $N_{{stat}}={}$".format(rt, nstat))
    ax.fill_between(tp, ave+err, ave-err, alpha=0.5, color=m.get_color())
ax.legend(loc="upper left", frameon=False)
ax.set_ylabel("I(t)")
ax.set_xlabel("t")
fig.canvas.draw()
fig.show()

fig = plt.figure(2)
fig.clf()
ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
for rt in rts:
    t = [sets[rt][x][:,0] for x in sets[rt]]
    R = [sets[rt][x][:,1] for x in sets[rt]]
    tp,yp = interpolate(t, R)
    nstat = yp.shape[0]
    ave = yp.mean(axis=0)
    err = yp.std(axis=0)/np.sqrt(yp.shape[0])
    m, = ax.plot(tp, ave, ls="-", lw=0.5, label=r"$d_f/a={:4.1f}$".format(rt))
    ax.fill_between(tp, ave+err, ave-err, alpha=0.5, color=m.get_color())
    ax.plot([rt, rt], [-0.1, max(ave)*1.1], ls="--", color=m.get_color())
    # ax.text(rt, max(ave)+0.1, "$d_f/a={:4.1f}$".format(rt), fontsize=11, va="bottom", ha="center")
# ax.legend(loc="upper left", frameon=False)
ax.set_ylabel("$\mathcal{R}(t)$")
ax.set_xlabel("$t/a$")
ax.set_xlim(0, 7)
ax.set_ylim(-0.1)
fig.canvas.draw()
fig.show()

fig = plt.figure(3)
fig.clf()
ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
data = list()
for rt in rts:
    t = [sets[rt][x][:,0] for x in sets[rt]]
    R = [sets[rt][x][:,1] for x in sets[rt]]
    ### Returns times between rt/2 and 3*rt/2
    tsl = [x[np.logical_and(rt/2 < x, x < 3*rt/2)] for x in t]
    ### Get the average step between consecutive times
    dt = np.array([(x[1:]-x[:-1]).mean() for x in tsl if len(x) > 1])
    dt = dt[np.isnan(dt) == False].mean()
    tp,yp = interpolate(t, R, xp=rt+dt)
    ave = yp.mean(axis=0)
    err = yp.std(axis=0)/np.sqrt(yp.shape[0])
    data.append((rt, ave, err))
x,ave,err = np.array(list(zip(*data)))
m,_,_ = ax.errorbar(x, ave, err, ls="", marker="o")
fit = lsqfit.nonlinear_fit(data=(x, gv.gvar(ave, err)), fcn=lambda x,p: p[0]*(x+p[1]), p0=[1,1])
x = np.linspace(min(x)-0.1, max(x))+0.1
y = fit.p[0]*(x+fit.p[1])
ax.plot(x, gv.mean(y), ls="-", color=m.get_color())
ax.fill_between(x, gv.mean(y)+gv.sdev(y), gv.mean(y)-gv.sdev(y), color=m.get_color(), alpha=0.2)
x = np.linspace(0, max(x)+1)
y = fit.p[0]*(x+fit.p[1])
ax.plot(x, gv.mean(y), ls="--", color=m.get_color())
ax.set_xlim(0)
ax.set_ylim(0)
ax.set_ylabel(r"$\mathcal{R}_0$=$\mathcal{R}$($d_f$)")
ax.set_xlabel(r"$d_f/a$")
fig.canvas.draw()
fig.show()

if False:
    plt.figure(2).savefig("Rt-scale-setting.tiff", dpi=300)
    plt.figure(3).savefig("R0-scale-setting.tiff", dpi=300)    
