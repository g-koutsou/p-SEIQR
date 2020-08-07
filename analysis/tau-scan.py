from matplotlib import pyplot as plt
from collections import defaultdict
import matplotlib as mpl
import numpy as np
import gvar as gv
import lsqfit
import tqdm
import re
import os

top = "data/tau-scan/"
lsdir = os.listdir(top)
rgxp = "out-seed([0-9]*)-tau{:4.2f}.txt"
taus = np.arange(1, 4.00+0.5, 0.25)
files = {tau: list(filter(lambda x: re.match(rgxp.format(tau), x), lsdir)) for tau in taus}

sets = dict()
for tau in tqdm.tqdm(taus):
    sets[tau] = defaultdict(list)
    for s,fname in enumerate(files[tau]):
        for line in open("{}/{}".format(top, fname), "r"):
            i,t,Rt,S,E,I,Q,RI,RQ = line.split()
            sets[tau][s,int(i)].append((float(t), float(Rt), int(S), int(E), int(I), int(Q), int(RI), int(RQ)))
    sets[tau] = {k: np.array(sets[tau][k]) for k in sets[tau]}

def interpolate(xs, ys, xp=None):
    if xp is None:
        maxx = np.concatenate(xs).max()
        minx = np.concatenate(xs).min()
        xp = np.linspace(minx, maxx, 256)
    yp = np.array([np.interp(xp, x, y) for x,y in zip(xs,ys)])
    return xp,yp

fig = plt.figure(1)
fig.clf()
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
for tau in taus:
    t = [sets[tau][x][:,0] for x in sets[tau]]
    I = [sets[tau][x][:,4] for x in sets[tau]]
    tp,yp = interpolate(t, I)
    nstat = yp.shape[0]
    ave = yp.mean(axis=0)
    err = yp.std(axis=0)/np.sqrt(yp.shape[0])
    m, = ax.plot(tp, ave, ls="-", lw=0.5, label=r"$\tau$={:4.2f}, $N_{{stat}}={}$".format(tau, nstat))
    ax.fill_between(tp, ave+err, ave-err, alpha=0.5, color=m.get_color())
ax.legend(loc="upper left", frameon=False)
ax.set_ylabel("I(t)")
ax.set_xlabel("t")
fig.canvas.draw()
fig.show()

fig = plt.figure(2)
fig.clf()
ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
for tau in taus[1::3]:
    t = [sets[tau][x][:,0] for x in sets[tau]]
    R = [sets[tau][x][:,1] for x in sets[tau]]
    tp,yp = interpolate(t, R)
    nstat = yp.shape[0]
    ave = yp.mean(axis=0)
    err = yp.std(axis=0)/np.sqrt(yp.shape[0])
    idx = tp >= tau*1.2
    m, = ax.plot(tp[idx], ave[idx], ls="-", lw=1, label=r"$\tau_i/a={:4.2f}$".format(tau))
    ax.fill_between(tp[idx], (ave+err)[idx], (ave-err)[idx], alpha=0.5, lw=0, color=m.get_color())
    # for x,y in zip(t,R):
    #     ax.plot(x, y, ls="-", lw=0.5, label=r"$\tau_i/a={:4.1f}$".format(tau))
    ax.plot([tau*6/5, tau*6/5], [-0.1, max(ave)*1.1], ls="--", color=m.get_color())
ax.set_ylabel("$R(t)$")
ax.legend(loc="upper left", frameon=False)
ax.set_xlabel("$t/a$")
ax.set_xlim(0, 6.5)
ax.set_ylim(0, 5.6)
fig.canvas.draw()
fig.show()

R0_target = 3.5

fig = plt.figure(3)
fig.clf()
ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
data = list()
for tau in taus:
    t = [sets[tau][x][:,0] for x in sets[tau]]
    R = [sets[tau][x][:,1] for x in sets[tau]]
    tp,yp = interpolate(t, R, xp=tau*6.01/5)
    ave = yp.mean(axis=0)
    err = yp.std(axis=0)/np.sqrt(yp.shape[0])
    data.append((tau, ave, err))
x,ave,err = np.array(list(zip(*data)))
m,_,_ = ax.errorbar(x, ave, err, ls="", marker="o")
fit = lsqfit.nonlinear_fit(data=(x, gv.gvar(ave, err)), fcn=lambda x,p: p[0]*(x+p[1]), p0=[1,1])
x = np.linspace(0, max(x)+3)
y = fit.p[0]*(x+fit.p[1])
ax.plot(x, gv.mean(y), ls="-", color=m.get_color())
ax.fill_between(x, gv.mean(y)+gv.sdev(y), gv.mean(y)-gv.sdev(y), color=m.get_color(), alpha=0.2)
#x = np.linspace(0, max(x)+1)
#y = fit.p[0]*(x+fit.p[1])
#ax.plot(x, gv.mean(y), ls="--", color=m.get_color())
if True:
    tau_i = R0_target/fit.p[0] - fit.p[1]
    a,e = gv.mean(tau_i),gv.sdev(tau_i)
    ax.plot([a]*2, [0, R0_target], ls="--", color="k")
    ax.plot([0,a], [R0_target]*2, ls="--", color="k")
    ax.fill_betweenx([0, R0_target], [a-e]*2, [a+e]*2, color="k", alpha=0.2, lw=0)
    ax.text(a+e, R0_target/2, r"$\tau_i/a = {}$".format(tau_i.fmt(2)), fontsize=9, ha="left", va="center")
ax.set_xlim(0, 6.5)
ax.set_ylim(0, 5.6)
ax.set_ylim(0)
ax.set_ylabel(r"${R}_0$=${R}$($\tau_i$+$\tau_e$)")
ax.set_xlabel(r"$\tau_i/a$")
fig.canvas.draw()
fig.show()

if False:
    for i,fn in [(2, "Rt-scale-setting.pdf"), (3,"R0-scale-setting.pdf"),]:
        fig = plt.figure(i)
        fig.patch.set_facecolor(None)
        fig.patch.set_alpha(0)
        fig.savefig(fn, facecolor=fig.get_facecolor())
