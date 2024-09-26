import os
import sys
import numpy as np
import subprocess
from ovito.io import *
from ovito.modifiers import *
from ovito.data import *
from ovito.pipeline import *
from ovito.vis import *

folder='/nfs/scistore15/saricgrp/acorso/actin/ivan/Test_B15__Alessandro'
filestringList = subprocess.check_output("ls {:s}/Results/Traj_rw_r0-*_Lz*_nF*-*_nM*_*.xyz | sed 's/.*Traj_//' | sed 's/.xyz//' ".format(folder), shell=True)
filestringList=filestringList.split()
filestringList=[a.decode("utf-8") for a in filestringList]

for filestring in filestringList:
    trajfilename  = '{:s}/Results/TrajSector_{:s}.xyz'.format(folder,filestring)
    bondsfilename   = '{:s}/Results/BondsSector_{:s}.dat'.format(folder,filestring)
    newbondsfilename   = '{:s}/Results/BondsSector_new_{:s}.dat'.format(folder,filestring)
    print(newbondsfilename)
    a = 0
    try: a = len(subprocess.check_output("ls "+newbondsfilename, shell=True,check=False).split())
    except ValueError:
        print("not found")
        pass
    if a > 0: 
        print("found")
        continue

    print(filestring)
    
    p = import_file(trajfilename, multiple_frames = True, sort_particles = True)

    fin = open(bondsfilename,'r')
    f = open(newbondsfilename, 'w')
    frames = p.source.num_frames
    for i in range(0,frames):
        data = p.compute(i)
        L = data.cell[0,0]
        Lz = data.cell[2,2]
        bonds = []
        conta = 0
        fin.readline()
        time = int(fin.readline().split()[0])
        fin.readline()
        totbonds = int(fin.readline().split()[0])
        fin.readline()
        fin.readline()
        fin.readline()
        fin.readline()
        fin.readline()
        for i in range(totbonds):
            bondline = fin.readline()
            if (int(bondline.split()[1]) in data.particles['Particle Identifier']) and (int(bondline.split()[2]) in data.particles['Particle Identifier']):
                bonds.append(bondline)
        f.write("ITEM: TIMESTEP\n")
        f.write("%d\n"%(data.attributes['Timestep']))
        f.write("ITEM: NUMBER OF ENTRIES\n")
        f.write("%d\n"%(len(bonds)))
        f.write("ITEM: BOX BOUNDS ff ff ff\n")
        f.write("-%.1f %.1f\n"%(L/2,L/2))
        f.write("-%.1f %.1f\n"%(L/2,L/2))
        f.write("-%.1f %.1f\n"%(Lz/2,Lz/2))
        f.write("ITEM: ENTRIES index c_cBonds[1] c_cBonds[2] c_cBonds[3] c_cBondsDet[1] c_cBondsDet[2]\n")
        for bondline in bonds:
            f.write(bondline)
    fin.close()
    f.close()