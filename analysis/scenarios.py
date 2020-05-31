import tqdm
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from collections import defaultdict
import gvar as gv
import lsqfit

rt = 4.38
seeds = list(range(1, 10))
vp = "A6"
qp = "A6"
pp = "A6"
Ni,Nq = 4,2

sets = {}
sets[vp] = defaultdict(list)
for s in seeds:
    if pp != "":
        fname = "data/scenarios/out-long-s{:02.0f}-t{:4.2f}-vp{}-qp{}-pp{}-Ni{:06.0f}-Nq{:06.0f}.txt".format(s,rt,vp,qp,pp,Ni,Nq)
    else:
        fname = "data/scenarios/out-long-s{:02.0f}-t{:4.2f}-vp{}-qp{}-Ni{:06.0f}-Nq{:06.0f}.txt".format(s,rt,vp,qp,Ni,Nq)
    for line in tqdm.tqdm(open(fname, "r"), desc="t={:4.2f}, seed={:d}".format(rt, s)):
        if len(line.split()) != 8:
            break
        i,t,Rt,S,I,Q,RI,RQ = line.split()
        sets[vp][s,int(i)].append((float(t), float(Rt), int(S), int(I), int(Q), int(RI), int(RQ)))
    if len(sets[vp]) != 1:
        sets[vp].pop((s,int(i)))
sets[vp] = {k: np.array(sets[vp][k]) for k in sets[vp]}

def interpolate(xs, ys, xp=None, npoints=256):
    if xp is None:
        maxx = np.concatenate(xs).max()
        minx = np.concatenate(xs).min()
        xp = np.linspace(minx, maxx, npoints)
    yp = np.array([np.interp(xp, x, y) for x,y in zip(xs,ys)])
    return xp,yp

cy_data = np.array([
          2,  2,  6, 10, 21, 26, 33,
         46, 49, 58, 67, 75, 84, 95,
        116,124,132,146,162,179,214,
        230,262,320,356,396,426,446,
        465,494,526,564,595,616,633,
        662,695,715,735,750,761,767,
        772,784,790,795,804,810,817,
        822,837,843,850,857,864,872,
        874,878,883,889,891,892,898
])

end_day = {"A4": 57,
           "A6": 60}[vp]

cy = cy_data[:end_day]

ascale = 10/rt # days per time unit

# fig = plt.figure(1, figsize=(7.5,5.4))
fig = plt.figure(1, figsize=(6,5))
fig.clf()
gs = mpl.gridspec.GridSpec(100, 100, left=0.15, right=0.975, bottom=0.1, top=0.975)
ax = fig.add_subplot(gs[:70,:])
t = [sets[vp][k][:,0] for k in sets[vp]]
R = [sets[vp][k][:,4]+sets[vp][k][:,6] for k in sets[vp]]
I = [sets[vp][k][:,3]+sets[vp][k][:,4] for k in sets[vp]]
tp,yp = interpolate(t, R)
ave = yp.mean(axis=0)
err = yp.std(axis=0)/np.sqrt(yp.shape[0])
m, = ax.plot(ascale*tp, ave, ls="-", lw=0.5)
ax.fill_between(ascale*tp, ave+err, ave-err, alpha=0.5, color=m.get_color(), label="Confirmed cases (cumulative)")
if True:    
    x,y = np.arange(len(cy)),cy
    ax.plot(x, y, ls="", color="k", marker="x", ms=3, label="Cyprus data (cumulative)")
if False:
    x,y = np.arange(len(cy)),cy
    N = y.shape[0]
    n = (N//7)*7
    x0 = x[:n].reshape([n//7,7]).mean(axis=1)
    y0 = y[:n].reshape([n//7,7]).mean(axis=1)
    ax.plot(x0, y0, ls="", color="k", marker="o", ms=5, label="Cyprus data (cumulative)\n")
    x1 = x[n:].mean()
    y1 = y[n:].mean()
    ax.plot(x1, y1, ls="", color="k", marker="o", ms=5)
if True:
    t = [sets[vp][k][:,0] for k in sets[vp]]
    I = [sets[vp][k][:,3] for k in sets[vp]]
    Q = [sets[vp][k][:,4] for k in sets[vp]]
    tp,yp = interpolate(t, Q)
    ave = yp.mean(axis=0)
    err = yp.std(axis=0)/np.sqrt(yp.shape[0])
    m, = ax.plot(ascale*tp, ave, ls="-", lw=0.5)
    ax.fill_between(ascale*tp, ave+err, ave-err, alpha=0.5, color=m.get_color(), label="Confirmed cases (active)")
    # tp,yp = interpolate(t, [i+q for i,q in zip(I,Q)])
    # ave = yp.mean(axis=0)
    # err = yp.std(axis=0)/np.sqrt(yp.shape[0])
    # m, = ax.plot(ascale*tp, ave, ls="-", lw=0.5)
    # ax.fill_between(ascale*tp, ave+err, ave-err, alpha=0.5, color=m.get_color(), label="Total cases (active)")
ax.legend(loc="upper left", frameon=False, ncol=1)
ax.set_ylabel(r"Infected")
ax.set_xticklabels([])
ax.set_ylim(0, 1750)
xmax = 172
ax.set_xticks(np.arange(10,xmax), minor=True)
#.
# ax = fig.add_subplot(gs[35:65,55:95])
# m, = ax.plot(ascale*tp, ave, ls="-", lw=0.5)
# ax.fill_between(ascale*tp, ave+err, ave-err, alpha=0.5, color=m.get_color(), label="Model: total confirmed")
# x,y = np.arange(len(cy["0"])),cy["0"]
# ax.plot(x, y, ls="", color="k", marker="x", ms=3, label="Cyprus data")
# ax.set_xlim(20,50)
# ax.set_ylim(100,880)
#.
ax = fig.add_subplot(gs[72:,:])
p = gv.gvar(["4.289(27)", "-0.0527(11)"])
ts,v = np.loadtxt("data/scenarios/vp{}.txt".format(vp)).T
if v.shape != ():
    widths=(ascale*(ts[1:]-ts[:-1])-0.5).tolist()
    widths.append(1e3)
else:
    widths=1000
R0 = p[0]*(v+p[1])
ba = ax.bar(ts*ascale, gv.mean(R0), align="edge", width=widths)
ax.plot([0, 1e3], [1,1], ls="--", color="k")
if pp != "":
    prms = gv.gvar(["3.995(15)", "-0.0295(52)", "-0.1388(17)", "0.00051(60)"])
    def z(p, x, y):
        a,b,c,d = p
        return gv.mean(a*x*y + b*x + c*y + d)
    probs = list()
    for line in open("data/scenarios/pp{}.txt".format(pp), "r"):
        if len(line.split()) > 2:
            t = float(line.split()[0])
            d = [t]
            for l in line.split()[1:]:
                pro,pop = list(map(float, l.split(":")))
                d += [pro,pop]
            probs.append(d)
    for i,pro in enumerate(probs):
        t0 = pro[0]
        tss = ts.tolist()+[1e3]
        for j,t in enumerate(tss[:-1]):
            if t0 >= t and t0 < tss[j+1]:
                vv = v[j]
        if i == len(probs)-1:
            t1 = 1e3
        else:
            t1 = probs[i+1][0]
        for pr,po in np.array(pro[1:]).reshape([-1,2]):
            if pr == 1.0:
                r0 = gv.mean(R0[j])
                ax.text(t0*ascale+2, r0-0.1, "{:2.0f}% of population".format(po*1e2), ha="left", va="top", fontsize=11)
            if pr != 1.0:
                r0 = z(prms, vv, pr)
                r,g,b,a = ba[0].get_facecolor()
                color = (r+0.2,g+0.2,b+0.2,a)
                ax.bar([tss[j]*ascale, tss[j+1]*ascale], [r0]*2, align="edge", width=widths[j], color=color)
                ax.text(t0*ascale+2, r0-0.1, "{:2.0f}% of population".format(po*1e2), ha="left", va="top", fontsize=11)
ts,rs = np.loadtxt("data/scenarios/qp{}.txt".format(qp)).T
if rs.shape == ():
    ax.text(0.5, 0.9, "{:2.0f} % of infected are detected".format(rs*1e2),
            ha="center", va="top", fontsize=11, transform=ax.transAxes)
else:
    for i,(t,r) in enumerate(zip(ts[:-1],rs[:-1])):
        y = max(gv.mean(R0[2:]))+.75
        ax.text(ascale*ts[i+1], y, r"{:2.0f}$\leftarrow$  ".format(r*1e2), va="top", ha="right", fontsize=11)
        ax.plot([ascale*ts[i+1]]*2, [y-.5,y], ls="-", color="k")
        ax.text(ascale*ts[i+1], y, r"$\rightarrow${:2.0f}".format(rs[i+1]*1e2), va="top", ha="left", fontsize=11)
    ax.text(ascale*ts[i+1], y, "                  % of infected detected",
            ha="left", va="top", fontsize=11)
ax.set_ylim(0, max(3.3, max(gv.mean(R0[2:]))+1.5))
# ax.set_ylim(0, max(3.3, max(gv.mean(R0[2:]))+1.5))
ax.set_ylim(0, 4.9)
ax.set_yticks(np.arange(0, int(ax.get_ylim()[1]+1)))
ax.set_ylabel(r"$\tilde{\mathcal{R}}_0$")
ax.set_xlabel("date")
#.
for ax in fig.axes[:]:
    ax.set_xlim(10,xmax)
ax = fig.axes[-1]
xt = ax.get_xticks()
fmt = lambda d: "/".join(list(reversed(str(d).split("-")))[:2])
xtls = [fmt(np.timedelta64(int(x), "[D]") + np.datetime64("2020-03-09")) for x in xt]
ax.set_xticklabels(xtls, fontsize=13)
ax.set_xticks(np.arange(10,xmax), minor=True)
fig.canvas.draw()
fig.show()

fig = plt.figure(2, figsize=(6.8, 3.2))
fig.clf()
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
t = [sets[vp][k][:,0] for k in sets[vp]]
I = [sets[vp][k][:,3] for k in sets[vp]]
Q = [sets[vp][k][:,4] for k in sets[vp]]
tp,yp = interpolate(t, Q)
ave = yp.mean(axis=0)
err = yp.std(axis=0)/np.sqrt(yp.shape[0])
m, = ax.plot(ascale*tp+4, ave, ls="-", lw=0.5)
ax.fill_between(ascale*tp+4, ave+err, ave-err, alpha=0.5, color=m.get_color(), label="Confirmed cases\n(detected)")
tp,yp = interpolate(t, [i+q for i,q in zip(I,Q)])
ave = yp.mean(axis=0)
err = yp.std(axis=0)/np.sqrt(yp.shape[0])
m, = ax.plot(ascale*tp+4, ave, ls="-", lw=0.5)
ax.fill_between(ascale*tp+4, ave+err, ave-err, alpha=0.5, color=m.get_color(), label="Total cases")
ax.legend(loc="upper right", frameon=False, ncol=8)
ax.set_ylabel(r"Infected")
ax.set_ylim(0, max(ave)*1.2)
ts,rs = np.loadtxt("data/scenarios/qp{}.txt".format(qp)).T
if rs.shape != ():
    for i,(t,r) in enumerate(zip(ts[:-1],rs[:-1])):
        ax.text(ascale*ts[i+1], max(ave)/3, r"{:2.0f}$\leftarrow$ ".format(r*1e2), va="top", ha="right", fontsize=11)
        # ax.plot([ascale*ts[i+1]]*2, [max(ave)/3,2*max(ave)/2], ls="-", color="k")
        x0,y0,dx,dy = ascale*ts[i+1],max(ave)/3,0,-max(ave)/3
        ax.arrow(x0, y0, dx, dy, length_includes_head=True, overhang=0.1,
                 width=0.5, head_width=3, head_length=60, lw=0, color="k")
        ax.text(ascale*ts[i+1], max(ave)/3, r"$\;\rightarrow${:2.0f}".format(rs[i+1]*1e2), va="top", ha="left", fontsize=11)
    ax.text(ascale*ts[i+1], max(ave)/3, "% of infected\ndetected",
            ha="center", va="bottom", fontsize=11)
xt = ax.get_xticks()
fmt = lambda d: "/".join(list(reversed(str(d).split("-")))[:2])
xtls = [fmt(np.timedelta64(int(x), "[D]") + np.datetime64("2020-03-09")) for x in xt]
ax.set_xticklabels(xtls, fontsize=13)
ax.set_xlabel("date")
fig.canvas.draw()
fig.show()

if False:
    plt.figure(1).savefig("{}-1.pdf".format(vp))
    plt.figure(2).savefig("{}-2.pdf".format(vp))

if False:
    plt.figure(1).savefig("{}-1.tiff".format(vp), dpi=300)
    plt.figure(2).savefig("{}-2.tiff".format(vp), dpi=300)
