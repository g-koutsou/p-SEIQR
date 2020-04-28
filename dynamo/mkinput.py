import xml.etree.ElementTree as ET
from xml.dom import minidom
from itertools import product
import numpy as np

np.random.seed(1)

Lx,Ly = 5000,5000
vel = 40
dims = {"x": "{}".format(Lx), "y": "{}".format(Ly), "z": "4"}
pos_x = np.arange(0,Lx,10)-Lx/2
pos_y = np.arange(0,Ly,10)-Ly/2
angles = np.random.rand(len(pos_x)*len(pos_y))*2*np.pi
vels_x = vel*np.cos(angles)
vels_y = vel*np.sin(angles)

tree = ET.Element("DynamOconfig", attrib={"version": "1.5.0"})
si = ET.SubElement(tree, "Simulation")
sc = ET.SubElement(si, "Scheduler", attrib={"Type": "NeighbourList"})
so = ET.SubElement(sc, "Sorter", attrib={"Type": "BoundedPQMinMax3"})
ss = ET.SubElement(si, "SimulationSize", attrib=dims)
ge = ET.SubElement(si, "Genus")
sp = ET.SubElement(ge, "Species", attrib={"Mass": "1", "Name": "Bulk", "Type": "Point"})
ir = ET.SubElement(sp, "IDRange", attrib={"Type": "All"})
bc = ET.SubElement(si, "BC", attrib={"Type": "PBC"})
to = ET.SubElement(si, "Topology")
nt = ET.SubElement(si, "Interactions")
ty = ET.SubElement(nt, "Interaction", attrib={"Type": "HardSphere", "Diameter": "1", "Name": "Bulk"})
ir = ET.SubElement(ty, "IDPairRange", attrib={"Type": "All"})
lo = ET.SubElement(si, "Locals")
go = ET.SubElement(si, "Globals")
se = ET.SubElement(si, "SystemEvents")
dt = ET.SubElement(si, "Dynamics", attrib={"Type": "Newtonian"})
pr = ET.SubElement(si, "Properties")
pd = ET.SubElement(tree, "ParticleData")
for i,(x,y) in enumerate(product(pos_x, pos_y)):
    vx,vy = vels_x[i],vels_y[i]
    pt = ET.SubElement(pd, "Pt", attrib={"ID": "{}".format(i)})
    po = ET.SubElement(pt, "P", attrib={"x": "{}".format(x), "y": "{}".format(y), "z": "0.0"})
    ve = ET.SubElement(pt, "V", attrib={"x": "{}".format(vx), "y": "{}".format(vy), "z": "0.0"})

flat = ET.tostring(tree, "utf-8")
frmt = minidom.parseString(flat)
print(frmt.toprettyxml(indent = "   "))
