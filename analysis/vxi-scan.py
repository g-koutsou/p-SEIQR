import tqdm
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from collections import defaultdict
from itertools import product
import gvar as gv
import lsqfit

rt = 4.38
vs = np.round(np.arange(0.05, 1.05, 0.05), 4)
pi = np.round(np.arange(0.05, 1.05, 0.05), 4)
seeds = list(range(1, 10))
ascale = 10/rt # days per time unit

sets = dict()
vxp = list(product(vs,pi))
for v,p in tqdm.tqdm(vxp):
    key = (v,p)
    sets[key] = defaultdict(list)
    for s in seeds:
        fname = "data/vxi-scan/out-s{:02.0f}-t{:4.2f}-v{:5.3f}-i{:5.3f}.txt".format(s,rt,v,p)
        for line in open(fname, "r"):
            i,t,Rt,S,I,Q,RI,RQ = line.split()
            sets[key][s,int(i)].append((float(t), float(Rt), int(S), int(I), int(Q), int(RI), int(RQ)))
    sets[key] = {k: np.array(sets[key][k]) for k in sets[key]}

def interpolate(xs, ys, xp=None, points=75):
    if xp is None:
        maxx = np.concatenate(xs).max()
        minx = np.concatenate(xs).min()
        xp = np.linspace(minx, maxx, points)
    yp = np.array([np.interp(xp, x, y) for x,y in zip(xs,ys)])
    return xp,yp

fig = plt.figure(1)
fig.clf()
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
v = 1
for p in pi:
    t = [sets[v,p][x][:,0] for x in sets[v,p]]
    I = [sets[v,p][x][:,3] for x in sets[v,p]]
    tp,yp = interpolate(t, I)
    nstat = yp.shape[0]
    ave = yp.mean(axis=0)
    err = yp.std(axis=0)/np.sqrt(yp.shape[0])
    m, = ax.plot(tp*ascale, ave, ls="-", lw=0.5, label=r"$p$={:5.3f}, $N_{{s}}$={}".format(p, nstat))
    ax.fill_between(tp*ascale, ave+err, ave-err, alpha=0.5, color=m.get_color())
ax.legend(loc="upper left", frameon=False, ncol=2)
ax.set_ylabel("I(t)")
ax.set_xlabel("t [days]")
fig.canvas.draw()
fig.show()

fig = plt.figure(2)
fig.clf()
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
for p in pi:
    t = [sets[v,p][x][:,0] for x in sets[v,p]]
    R = [sets[v,p][x][:,1] for x in sets[v,p]]
    tp,yp = interpolate(t, R)
    nstat = yp.shape[0]
    ave = yp.mean(axis=0)
    err = yp.std(axis=0)/np.sqrt(yp.shape[0])
    m, = ax.plot(tp*ascale, ave, ls="-", lw=0.5, label=r"$p$={:5.3f}, $N_{{s}}$={}".format(p, nstat))
    ax.fill_between(tp*ascale, ave+err, ave-err, alpha=0.5, color=m.get_color())
    ax.plot([rt*ascale, rt*ascale], [0, max(ave)], ls="--", color=m.get_color())
ax.legend(loc="upper left", frameon=False, ncol=3)
ax.set_ylim(0, 4)
ax.set_xlim(0,28)
ax.set_ylabel("R(t)")
ax.set_xlabel("t [days]")
fig.canvas.draw()
fig.show()

R0 = np.zeros([len(vs), len(pi), 2])
for i,v in enumerate(vs):
    for j,p in enumerate(pi):
        t = [sets[v,p][x][:,0] for x in sets[v,p]]
        R = [sets[v,p][x][:,1] for x in sets[v,p]]
        ### Returns times between rt/2 and 3*rt/2
        tsl = [x[np.logical_and(rt/2 < x, x < 3*rt/2)] for x in t]
        ### Get the average step between consecutive times
        dt = np.array([(x[1:]-x[:-1]).mean() for x in tsl if len(x) > 1])
        dt = dt[np.isnan(dt) == False].mean()
        tp,yp = interpolate(t, R, xp=rt+dt, points=256)
        ave = yp.mean(axis=0)
        err = yp.std(axis=0)/np.sqrt(yp.shape[0])
        R0[i,j,:] = np.array([ave, err])

fits = defaultdict(dict)

fig = plt.figure(3)
fig.clf()
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
for i,v in enumerate(vs):
    ave,err = R0[i,:,:].T
    x = pi
    m,_,_ = ax.errorbar(x, ave, err, ls="", marker="o", label="$v={:f}$".format(v))
    fit = lsqfit.nonlinear_fit(data=(x, gv.gvar(ave, err)), fcn=lambda x,p: p[0]*(x+p[1]), p0=[1,1])
    x = np.linspace(min(x)-0.01, max(x)+0.01)
    y = fit.p[0]*(x+fit.p[1])
    ax.plot(x, gv.mean(y), ls="-", color=m.get_color())
    ax.fill_between(x, gv.mean(y)+gv.sdev(y), gv.mean(y)-gv.sdev(y), color=m.get_color(), alpha=0.2)
    x = np.linspace(0, max(x)+1)
    y = fit.p[0]*(x+fit.p[1])
    ax.plot(x, gv.mean(y), ls=":", color=m.get_color())
    fits["v"][v] = fit
ax.legend(loc="upper left", frameon=False, ncol=1)
ax.set_xlim(0, 1.1)
ax.set_ylim(0, 4.5)
ax.set_ylabel(r"$R_0$=R($\tau$)")
ax.set_xlabel(r"$p$")
fig.canvas.draw()
fig.show()

fig = plt.figure(4)
fig.clf()
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
for p in [0.25, 0.5, 0.75]:
    j = pi.tolist().index(p)
    ave,err = R0[:,j,:].T
    x = vs
    m,_,_ = ax.errorbar(x, ave, err, ls="", marker="o", label="$p={:g}$".format(p))
    fit = lsqfit.nonlinear_fit(data=(x, gv.gvar(ave, err)), fcn=lambda x,p: p[0]*(x+p[1]), p0=[1,1])
    x = np.linspace(min(x)-0.01, max(x)+0.01)
    y = fit.p[0]*(x+fit.p[1])
    ax.plot(x, gv.mean(y), ls="-", color=m.get_color())
    ax.fill_between(x, gv.mean(y)+gv.sdev(y), gv.mean(y)-gv.sdev(y), color=m.get_color(), alpha=0.2)
    x = np.linspace(0, max(x)+1)
    y = fit.p[0]*(x+fit.p[1])
    ax.plot(x, gv.mean(y), ls=":", color=m.get_color())
    fits["p"][p] = fit
ax.legend(loc="upper left", frameon=False, ncol=1)
ax.set_xlim(0, 1.1)
ax.set_ylim(0, 3.45)
ax.set_ylabel(r"$R_0$")
ax.set_xlabel(r"$v/v_0$")
fig.canvas.draw()
fig.show()

def fcn(xy, p):
    x,y = xy
    a,b,c,d = p
    return a*x*y + b*x + c*y + d

xy = np.array(list(product(vs, pi))).T
fit = lsqfit.nonlinear_fit(data=(xy, gv.gvar(R0[:,:,0], R0[:,:,1])), fcn=fcn, p0=[1,]*4)

fig = plt.figure(5)
fig.clf()
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
levels = np.logspace(-1,0.5,10)
ct = ax.contour(vs, pi, R0[:,:,0], levels=levels)
ax.clabel(ct, inline=1, fontsize=9, fmt="%4.2f")
z = fcn(xy, gv.mean(fit.p)).reshape(R0[:,:,0].shape)
ct = ax.contour(vs, pi, z, levels=levels, linestyles="--")
ax.set_ylabel("p")
ax.set_xlabel("$v/v_0$")
fig.canvas.draw()
fig.show()

if True:
    for i in [1,2,3,4,5]:
        plt.figure(i).savefig("vxi-scan-{}.pdf".format(i), bbox_inches="tight") 
