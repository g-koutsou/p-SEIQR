import multiprocess
import numpy as np
import subprocess
import itertools
import tqdm

cmd = "./p-SEIQR/c-SIR/c-sir"
trajs = "./data/trajs/traj-tail-n00250000.txt"

Npop = 250000
E0 = 1

tau_i = 3.617
tau_e = 0.2*tau_i
def func(v, p, seed, NT):
    c = list()
    c.append(r"{}".format(cmd))
    c.append(r"-s {}".format(seed))
    c.append(r"-E {}".format(E0))
    c.append(r"-t {}".format(tau_i))
    c.append(r"-e {}".format(tau_e))
    c.append(r"-T {}".format(NT))
    c.append(r"-v {}".format(v))
    c.append(r"-i {}".format(p))
    c.append(r"{}".format(Npop))
    c.append(r"{}".format(trajs))
    r = subprocess.run(c, stdout=subprocess.PIPE)
    d = np.array(list(map(str.split, r.stdout.decode().split("\n")[:-1])), float)
    fname = "out-seed{:05.0f}-vel{:6.4f}-pr{:6.4f}-tau{:5.3f}.txt".format(seed, v, p, tau_i)
    np.savetxt("data/vel-scan/{}".format(fname), d, fmt=["%4d", "%12.6f", "%8.5f"] + ["%12d"]*6)
    return 

T = 6
dv = 0.125
dp = 0.100
vels = np.arange(0.025, 2+dv, dv)
prbs = np.arange(0.100, 1+dp, dp)
vp = itertools.product(vels, prbs)
cycle = itertools.cycle(vp)
prms = list()
for seed in range(len(vels)*len(prbs)*128):
    v,p = next(cycle)
    prms.append((v, p, seed, T))

pool = multiprocess.Pool(None)
r = pool.starmap_async(func, prms)
r.wait()
pool.close()

