from matplotlib import pyplot as plt
from collections import defaultdict
import matplotlib as mpl
import numpy as np
import gvar as gv
import itertools
import lsqfit
import tqdm
import os
import re

top = "data/vel-scan/"
lsdir = os.listdir(top)
rgxp = "out-seed([0-9]*)-vel{:6.4f}-pr{:6.4f}-tau{:5.3f}.txt"
tau_i = 3.617
dv = 0.125
dp = 0.1
vels = np.arange(0.15, 2+dv, dv)
prbs = np.arange(0.1, 1+dp, dp)
files = {(vel,prb): list(filter(lambda x: re.match(rgxp.format(vel, prb, tau_i), x), lsdir)) for vel,prb in itertools.product(vels,prbs)}

tau_e = tau_i*0.2
tau = tau_i+tau_e
ascale = 10/tau_i # days per time unit

sets = dict()
for vel,prb in tqdm.tqdm(list(itertools.product(vels,prbs))):
    sets[vel,prb] = defaultdict(list)
    for s,fname in enumerate(files[vel,prb]):
        for line in open("{}/{}".format(top, fname), "r"):
            i,t,Rt,S,E,I,Q,RI,RQ = line.split()
            sets[vel,prb][s,int(i)].append((float(t), float(Rt), int(S), int(I), int(Q), int(RI), int(RQ)))
    sets[vel,prb] = {k: np.array(sets[vel,prb][k]) for k in sets[vel,prb]}

def interpolate(xs, ys, xp=None):
    if xp is None:
        maxx = np.concatenate(xs).max()
        minx = np.concatenate(xs).min()
        xp = np.linspace(minx, maxx, 8192)
    yp = np.array([np.interp(xp, x, y) for x,y in zip(xs,ys)])
    return xp,yp

fig = plt.figure(1)
fig.clf()
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
for p in [1,0.8,0.6,0.4]:#prbs:
    data = list()
    for v in vels:
        t = [sets[v,p][x][:,0] for x in sets[v,p]]
        R = [sets[v,p][x][:,1] for x in sets[v,p]]
        ### Returns times between tau/2 and 3*tau/2
        tsl = [x[np.logical_and(tau-ascale/10 < x, x < tau+ascale/10)] for x in t]
        ### Get the average step between consecutive times
        dt = np.array([(x[1:]-x[:-1]).mean() for x in tsl if len(x) > 1])
        dt = dt[np.isnan(dt) == False].mean()
        tp,yp = interpolate(t, R, xp=tau)
        ave = yp.mean(axis=0)
        err = yp.std(axis=0)/np.sqrt(yp.shape[0])
        data.append((v, ave, err))
    x,ave,err = np.array(list(zip(*data)))
    m,_,_ = ax.errorbar(x, ave, err, ls="", marker="o", label="p={:5.1f}".format(p))
    fit = lsqfit.nonlinear_fit(data=(x, gv.gvar(ave, err)), fcn=lambda x,p: p[0]*(x+p[1]), p0=[1,1])
    x = np.linspace(min(x)-0.01, max(x)+0.01)
    y = fit.p[0]*(x+fit.p[1])
    ax.plot(x, gv.mean(y), ls="-", color=m.get_color())
    ax.fill_between(x, gv.mean(y)+gv.sdev(y), gv.mean(y)-gv.sdev(y), color=m.get_color(), alpha=0.2)
    x = np.linspace(0, max(x)+1)
    y = fit.p[0]*(x+fit.p[1])
    ax.plot(x, gv.mean(y), ls=":", color=m.get_color())
    print("{0:4.2f}   {1[0]:<12s} {1[1]:<12s} {2:4.2f}".format(p, fit.p, fit.chi2/fit.dof))
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.set_xlim(0, 2.1)
ax.set_ylim(0, 8.2)
ax.set_ylabel(r"$R_0$=R($\tau_i$+$\tau_e$)")
ax.set_xlabel(r"$u/v_0$")
fig.canvas.draw()
fig.show()

if False:
    fn = "vel-prob-scan.pdf"
    fig.patch.set_facecolor(None)
    fig.patch.set_alpha(0)
    fig.savefig(fn, facecolor=fig.get_facecolor())

fig = plt.figure(2)
fig.clf()
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
for v in vels:
    data = list()
    for p in prbs:
        t = [sets[v,p][x][:,0] for x in sets[v,p]]
        R = [sets[v,p][x][:,1] for x in sets[v,p]]
        ### Returns times between tau/2 and 3*tau/2
        tsl = [x[np.logical_and(tau-ascale/10 < x, x < tau+ascale/10)] for x in t]
        ### Get the average step between consecutive times
        dt = np.array([(x[1:]-x[:-1]).mean() for x in tsl if len(x) > 1])
        dt = dt[np.isnan(dt) == False].mean()
        tp,yp = interpolate(t, R, xp=tau)
        ave = yp.mean(axis=0)
        err = yp.std(axis=0)/np.sqrt(yp.shape[0])
        data.append((p, ave, err))
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
    print("{0:4.2f}   {1[0]:<12s} {1[1]:<12s} {2:4.2f}".format(v, fit.p, fit.chi2/fit.dof))
ax.set_xlim(0, 1.08)
ax.set_ylim(0, 8.2)
ax.set_ylabel(r"$R_0$=R($\tau_i$+$\tau_e$)")
ax.set_xlabel(r"$p$")
fig.canvas.draw()
fig.show()

data = list()
for v in vels:
    for p in prbs:
        t = [sets[v,p][x][:,0] for x in sets[v,p]]
        R = [sets[v,p][x][:,1] for x in sets[v,p]]
        ### Returns times between tau/2 and 3*tau/2
        tsl = [x[np.logical_and(tau-ascale/10 < x, x < tau+ascale/10)] for x in t]
        ### Get the average step between consecutive times
        dt = np.array([(x[1:]-x[:-1]).mean() for x in tsl if len(x) > 1])
        dt = dt[np.isnan(dt) == False].mean()
        tp,yp = interpolate(t, R, xp=tau)
        ave = yp.mean(axis=0)
        err = yp.std(axis=0)/np.sqrt(yp.shape[0])
        data.append((v, p, ave, err))

def R0(x, p):
    vel,pro = np.array(x)
    #return vel*pro*p[0] + vel*p[1] + pro*p[2] + p[3]
    return p[0]*vel*pro + p[1]*vel + p[2]*pro + p[3]
vel,pro,ave,err = zip(*data)
fit = lsqfit.nonlinear_fit(data=((vel, pro), gv.gvar(ave, err)), fcn=R0, p0=[1]*4)
        
if False:
    for i in [1,2,3]:
        plt.figure(i).savefig("v-scan-{}.pdf".format(i), bbox_inches="tight") 
