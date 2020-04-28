import numpy as np
import argparse
import tqdm
import sys
import re

parser = argparse.ArgumentParser()
parser.add_argument("fname", metavar="FNAME", type=str, help="input filename")
parser.add_argument("-o", dest="output", default=None, help="output filename (default: stdout)")
args = parser.parse_args(sys.argv[1:])
fname = args.fname
output = args.output

m0 = re.compile(".*t=(.*),.*")
m1 = re.compile(".*p1=([0-9]*), p2=([0-9]*),.*")

particle,times = list(),list()
with open(fname, "r") as fp:
    i = 0
    next_line = False
    for line in tqdm.tqdm(fp, leave=True):
        if next_line:
            p0,p1 = list(map(int, m1.match(line).groups()))
            particle.append((p0, p1))
            next_line = False
        if "Event Type=CORE" in line:
            t, = list(map(float, m0.match(line).groups()))
            times.append(t)
            next_line = True
        i += 1

arr = np.concatenate([np.array([times]).T, np.array(particle)], axis=1)
if output is not None:
    np.savetxt("{}".format(output), arr[:, :], fmt="%12.6f %12d %12d")
else:
    for row in arr:
        print("{:18.4f} {:12.0f} {:12.0f}".format(*row))
