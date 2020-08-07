import multiprocessing
import numpy as np
import subprocess
import itertools
import tqdm

cmd = "./p-SIR/c-SIR/c-sir"
trajs = "./data/trajs/traj-tail-n00250000.txt"

Npop = 250000
E0 = 1

def func(tau_i, seed, NT):
    tau_e = 0.2*tau_i
    c = list()
    c.append(r"{}".format(cmd))
    c.append(r"-s {}".format(seed))
    c.append(r"-E {}".format(E0))
    c.append(r"-t {}".format(tau_i))
    c.append(r"-e {}".format(tau_e))
    c.append(r"-T {}".format(NT))
    c.append(r"{}".format(Npop))
    c.append(r"{}".format(trajs))
    r = subprocess.run(c, stdout=subprocess.PIPE)
    d = np.array(list(map(str.split, r.stdout.decode().split("\n")[:-1])), float)
    return (tau_i, seed), d

prms = list()
taus = np.arange(1., 4.25+0.25, 0.25)
cycle = itertools.cycle(taus)
for seed in range(len(taus)*128,len(taus)*256):
    tau_i = next(cycle)
    prms.append((tau_i, seed, tau_i*1.5))

pool = multiprocessing.Pool(None)
r = pool.starmap_async(func, prms)
r.wait()
r = r.get()
pool.close()

for x in tqdm.tqdm(r):
    (tau_i,seed),d = x
    fname = "out-seed{:05.0f}-tau{:4.2f}.txt".format(seed, tau_i)
    np.savetxt("data/tau-scan/{}".format(fname), d, fmt=["%4d", "%12.6f", "%8.5f"] + ["%12d"]*6)

