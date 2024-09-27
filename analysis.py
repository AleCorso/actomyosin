import numpy as np
import pandas as pd
import os
import subprocess
import re
import pickle
#from numba import njit
#import time

#import PySide6.QtWidgets
#app = PySide6.QtWidgets.QApplication() # Needed to print ovito renderings on screen

from ovito.io import *
from ovito.modifiers import *
from ovito.data import *
from ovito.pipeline import *
from ovito.vis import *
#import PySide6.QtCore
import os.path

def getparams(pattern):
    r1 = re.compile('[_]')
    r2 = re.compile(r'(([-+]?\d+\.\d+)|([-+]?\d+))')
    r3 = re.compile('[-]')
    pattern = pattern.split()[0]
    #pattern = pattern.decode("utf-8")
    
    # Default params
    ts=-1
    real=-1
    Rcylin=-1
    Rcylout=-1
    Rring = -1
    geometry =''
    Lz=-1
    RN = 0
    pAN = 0
    pNN = 0
    pP = 0
    pB = 0
    pS = 0
    pD = 0
    pMA = 0
    pMW = 0
    pMD = 0
    pMDE = 0
    pCA=0
    pCD=0
    pMOC = 0
    nM=0
    nC=0
    eAA=0
    Lx=0
    
    # Read params
    for s in r1.split(pattern):
        if s.startswith('rw'):
            geometry = 'rw'
        elif s.startswith('cw'):
            geometry = 'cw'
        elif s.startswith('r'):
            s3 = s[1:]
            Rcylin = float(r3.split(s3)[0])
            Rcylout = float(r3.split(s3)[1])
            try:
                Rring = float(r3.split(s3)[2])
            except:
                pass
        if s.startswith('Lz'):
            Lz = float(r2.split(s)[1])
        if s.startswith('Lx'):
            Lx = float(r2.split(s)[1])    
        if s.startswith('eAA'):
            eAA = float(r2.split(s)[1])
        if s.startswith('nF'):
            s3 = s[2:]
            nF = int(r3.split(s3)[0])
            lF = int(r3.split(s3)[1])
        if s.startswith('nM'):
            nM = int(r2.split(s)[1])
        if s.startswith('nC'):
            nC = int(r2.split(s)[1])
        if s.startswith('R'):
            s3 = s[1:]
            RN = int(r3.split(s3)[0])
            pAN = float(r3.split(s3)[1])
            pNN = float(r3.split(s3)[2])
            pP = float(r3.split(s3)[3])
            pB = float(r3.split(s3)[4])
            pS = float(r3.split(s3)[5])
            pD = float(r3.split(s3)[6])
            pMA = float(r3.split(s3)[7])
            pMW = float(r3.split(s3)[8])
            pMD = float(r3.split(s3)[9])
            pMDE = float(r3.split(s3)[10])  
            pCA =float(r3.split(s3)[11])
            pCD =float(r3.split(s3)[12])
            pMOC = float(r3.split(s3)[13])  
        if s[0].isdigit():
            real=int(s)
    
    return {'geom':geometry, 'Rcylin':Rcylin, 'Rcylout':Rcylout, 'Rring':Rring, 'Lz':Lz, 'eAA':eAA, 'nF':nF, 'lF':lF, 'nM':nM, 'nC':nC, 'RN':RN, 
            'pAN':pAN,'pNN':pNN,'pP':pP,'pB':pB,'pS':pS,'pD':pD,'pMA':pMA,'pMW':pMW,'pMD':pMD,'pMDE':pMDE,'pCA':pCA,'pCD':pCD,'pMOC':pMOC,
            'real':real}

def sortbyparams(key):
    p = getparams(key)
    return 1e5*p['lF']+p['nM']

def nM2color(nM):
    if nM==19200:
        return 'black'
    if nM==9600:
        return 'navy'
    elif nM==4800:
        return 'royalblue'
    elif nM==2400:
        return 'teal'
    elif nM==1200:
        return 'forestgreen'
    elif nM==600:
        return 'green'
    elif nM==300:
        return 'lime'
    else:
        assert False, 'nM={} not in color palette nM2color'.format(nM)
    
def bin2midpoint(edges):
    return [0.5*(edges[i]+edges[i+1]) for i in range(0,len(edges)-1)]
#@njit
def ThetaDiff(th1,th2):
    a = abs(th1-th2)
    if a<np.pi:
        return a
    else:
        return 2*np.pi-a

#@njit
def ThetaOverlap(Th1,Th2,ThetaTolerance):
    for th1 in Th1:
        for th2 in Th2:
            if ThetaDiff(th1,th2)<ThetaTolerance:
                return True
    return False

#@njit
def zDiffAbs(z1,z2,Lz):
    a = abs(z1-z2)
    if a<0.5*Lz:
        return a
    else:
        return Lz-a
    
#@njit
def Distance2_ffp(p,q,Lz):
    assert len(p)==3
    assert len(q)==3
    a = (p[0]-q[0])**2 + (p[1]-q[1])**2 + (zDiffAbs(p[2],q[2],Lz))**2
    return a

#@njit
def zDiffSigned(z1,z2,Lz):
    a = (z1-z2)
    if abs(a)<0.5*Lz:
        return a
    elif a<-0.5*Lz:
        return a+Lz
    else:
        return a-Lz

#@njit
def zDiffSigned0(a,Lz):
    if abs(a)<0.5*Lz:
        return a
    elif a<-0.5*Lz:
        return a+Lz
    else:
        return a-Lz
    

# Traj analysis

folder='/nfs/scistore15/saricgrp/acorso/actin/ivan/Test_B15__Alessandro'
MMbondLength =10
oldfiles = subprocess.check_output("ls {:s}/Results/Traj_rw_r0-150*_Lz*_nF*-*_nM*_2[0-1][0-9].xyz | sed 's/.*Traj_//' | sed 's/.xyz//' ".format(folder), shell=True)
oldfiles=oldfiles.split()
oldfiles=[a.decode("utf-8") for a in oldfiles]
middlefiles = subprocess.check_output("ls {:s}/Results/Traj_rw_r0-250*_Lz*_nF*-*_nM*_203.xyz | sed 's/.*Traj_//' | sed 's/.xyz//' ".format(folder), shell=True)
middlefiles=middlefiles.split()
middlefiles=[a.decode("utf-8") for a in middlefiles]
finefiles = subprocess.check_output("ls {:s}/Results/Traj_rw_r0-*_Lz*_nF*-*_nM*_205.xyz | sed 's/.*Traj_//' | sed 's/.xyz//' ".format(folder), shell=True)
finefiles=finefiles.split()
finefiles=[a.decode("utf-8") for a in finefiles]
longfiles = subprocess.check_output("ls {:s}/Results/Traj_rw_r0-*_Lz*_nF*-*_nM*_214.xyz | sed 's/.*Traj_//' | sed 's/.xyz//' ".format(folder), shell=True)
longfiles=longfiles.split()
longfiles=[a.decode("utf-8") for a in longfiles]
lotfiles = subprocess.check_output("ls {:s}/Results/Traj_rw_r0-*_Lz*_nF*-*_nM*_204.xyz | sed 's/.*Traj_//' | sed 's/.xyz//' ".format(folder), shell=True)
lotfiles=lotfiles.split()
lotfiles=[a.decode("utf-8") for a in lotfiles]

# Look for all Traj files

filestringList = subprocess.check_output("ls {:s}/Results/Traj_rw_r0-*_Lz*_nF*-*_nM*_2[0-1][0-9].xyz | sed 's/.*Traj_//' | sed 's/.xyz//' ".format(folder), shell=True)
#filestringList = subprocess.check_output("ls {:s}/Results/Traj_rw_r0-*_Lz*_nF*-*_nM*_212.xyz | sed 's/.*Traj_//' | sed 's/.xyz//' ".format(folder), shell=True)
#filestringList = subprocess.check_output("ls {:s}/Results/Traj_cw_r*_Lz*_nF*-*_nM*_1[1-9][0-9].xyz | sed 's/.*Traj_//' | sed 's/.xyz//' ".format(folder), shell=True)
filestringList=filestringList.split()
filestringList=[a.decode("utf-8") for a in filestringList]

for f in filestringList:
    print(f)
    print(getparams(f))

# read dictionary 
#with open('{:s}/Analysis/dicR.pickle'.format(folder), 'rb') as file:
dicR={}
with open('/nfs/scistore15/saricgrp/acorso/actin/ivan/Test_B15__Alessandro/Analysis/dicR_05_09.pickle','rb') as file:
    dicR=pickle.load(file)
dicR.keys()
def addLocalFilamentDirection(dfm, Lz):
    assert len(dfm.Molec.unique())==1 # all atoms belong to same filament
    dfm = dfm.sort_values(by='PolyTime',ascending=True)
    Directionx = dfm.x.values[:-1]-dfm.x.values[1:]          # Direction goes from Barbed to Pointed end
    Directiony = dfm.y.values[:-1]-dfm.y.values[1:]
    Directionz = dfm.z.values[:-1]-dfm.z.values[1:]
    Directionz = [ zDiffSigned0(z,Lz) for z in Directionz ]
    Direction = np.array([Directionx, Directiony, Directionz]).transpose()
    Direction = np.append(Direction,[Direction[-1]],axis=0)
    Direction = [np.array(d) for d in Direction]
    dfm['Direction'] = Direction
    return dfm


def globalFilamentDirection(dfm, Lz):
    assert len(dfm.Molec.unique())==1 # all atoms belong to same filament
    dfm = dfm.sort_values(by='PolyTime',ascending=True)
    Directionx = dfm.x.values[0]-dfm.x.values[-1]          # Direction goes from Barbed [-1] to Pointed end [0]
    Directiony = dfm.y.values[0]-dfm.y.values[-1]
    Windingz = 0
    z = dfm.z.values
    Lzhalf = 0.5*Lz
    for k in range(len(z)-1):
        if z[k]>Lzhalf and z[k+1]<-Lzhalf:
            Windingz += 1
        elif z[k]<-Lzhalf and z[k+1]>Lzhalf:
            Windingz -= 1
    Directionz = dfm.z.values[0]-(dfm.z.values[-1]+(Windingz*Lz))
    return np.array([Directionx, Directiony, Directionz])


#@njit
def countcontacts(dm1,dm2,Distance2Tolerance,ThetaScreeningTolerance,Dir1,Dir2,lF):
    ContactDummyCounter = 0
    ContactCounter = [0,0,0]
    for ja1, a1 in enumerate(dm1):
        for ja2, a2 in enumerate(dm2):
            if ThetaDiff(a1[4],a2[4]) < ThetaScreeningTolerance :     # [4] is ThetaRad
                if Distance2_ffp([a1[1],a1[2],a1[3]],[a2[1],a2[2],a2[3]],Lz) < Distance2Tolerance :     # [1] is x, [2] is y, [3] is z
                    ContactDummyCounter += 1
                    break
    if ContactDummyCounter ==0:
        return ContactCounter
    if np.sign( np.dot(Dir1,Dir2) ) >= 0:    # [5] is Directionx, [6] is Directiony, [7] is Directionz
        ContactFlag = 0    # Parallel 
    else:
        p1 = dm1[0]  # Pointed end of molecule 1
        b1 = dm1[-1] # Barbed end of molecule 1
        p2 = dm2[0]  # Pointed end of molecule 2
        b2 = dm2[-1] # Barbed end of molecule 2
        if Distance2_ffp([p1[1],p1[2],p1[3]],[p2[1],p2[2],p2[3]],Lz) > Distance2_ffp([b1[1],b1[2],b1[3]],[b2[1],b2[2],b2[3]],Lz) : # distance between pointed ends > barbed ends
            ContactFlag=1    # AntiParallelExtensile
            #print("extensile")
        else:
            ContactFlag=2    # AntiParallelContractile
            #print("contractile")
    ContactCounter[ContactFlag] += ContactDummyCounter     # ContactCounter only counts contacts of molecule 1
    return ContactCounter

def MMbond_boundtosamefilaments(dfBondMM,dfBondMM_Reference):
    return dfPart.loc[dfBondMM.TopologyA]['MolecA_Mpart']
#@njit
def is_permuation_matrix(x):
    return (x.ndim == 2 and x.shape[0] == x.shape[1] and
            (x.sum(axis=0) == 1).all() and 
            (x.sum(axis=1) == 1).all() and
            ((x == 1) | (x == 0)).all())

#@njit
def which_indeces_merge(Matrix):
    if is_permuation_matrix(Matrix):
        return np.array([0,0,-1],dtype=np.int64) # do not sum
    for i in range(Matrix.shape[0]):
            for j1 in range(Matrix.shape[1]):
                for j2 in range(j1+1, Matrix.shape[1]):
                    if Matrix[i,j1]*Matrix[i,j2]==1:
                        return np.array([j1,j2,1],dtype=np.int64) # sum columns
    for j in range(Matrix.shape[1]):
            for i1 in range(Matrix.shape[0]):
                for i2 in range(i1+1, Matrix.shape[0]):
                    if Matrix[i1,j]*Matrix[i2,j]==1:
                        return np.array([i1,i2,0],dtype=np.int64) # sum rows 
    return np.array([0,0,-2],dtype=np.int64) # error

def mergeclusters(AdjacencyMatrix):
    clusterdic = 0*AdjacencyMatrix
    Matrix = AdjacencyMatrix
    # Initialise tracking
    Mx, My = Matrix.shape
    Tracker = [ [ [[i,j]] for j in range(My)] for i in range(Mx) ]
    for i in range(Mx):
        for j in range(My):
            if AdjacencyMatrix[i,j]==0:
                Tracker[i][j]=[]
    # Reduce matrix
    indeces = which_indeces_merge(Matrix)
    while(indeces[2]>=0):
        Mx, My = Matrix.shape
        #print(indeces)
        if indeces[2]==1: # sum columns, ie multiply Matrix~(Mx,My) by Mult~(My,My-1) to get NewMatrix~(Mx,My-1) 
            j1, j2 = indeces[:2]
            Mult = np.zeros( (My, My-1), dtype=int)
            for j in range(My):
                if j<j2:
                    Mult[j,j]=1
                elif j==j2:
                    Mult[j2,j1]=1
                else:
                    Mult[j,j-1]=1
            NewMatrix = np.sign(np.dot(Matrix,Mult)) # performs logic OR of columns j1 and j2
            # Update tracker
            '''NewTracker = [ [[] for j in range(My-1)] for i in range(Mx)] 
            for i in range(Mx):
                for j in range(My):
                    if j<j2:
                        NewTracker[i][j] += Tracker[i][j]
                    elif j==j2:
                        NewTracker[i][j1] += Tracker[i][j2]
                    else:
                        NewTracker[i][j-1] += Tracker[i][j]'''
            NewTracker = [[ Tracker[i][j] if (j<j2 and j!=j1)   else   Tracker[i][j1]+Tracker[i][j2] if j==j1   else   Tracker[i][j+1]   for j in range(My-1)] for i in range(Mx)]
            Matrix = NewMatrix
            Tracker = NewTracker
        
        if indeces[2]==0: # sum rows, ie multiply Mult~(Mx-1,Mx) Matrix~(Mx,My) to get NewMatrix~(Mx-1,My) 
            i1, i2 = indeces[:2]
            Mult = np.zeros( (Mx-1, Mx), dtype=int)
            for i in range(Mx):
                if i<i2:
                    Mult[i,i]=1
                elif i==i2:
                    Mult[i1,i2]=1
                else:
                    Mult[i-1,i]=1
            NewMatrix = np.sign(np.dot(Mult,Matrix))
            # Update tracker
            '''NewTracker = [ [[] for j in range(My)] for i in range(Mx-1)] 
            for i in range(Mx):
                for j in range(My):
                    if i<i2:
                        NewTracker[i][j] += Tracker[i][j]
                    elif i==i2:
                        NewTracker[i1][j] += Tracker[i2][j]
                    else:
                        NewTracker[i-1][j] += Tracker[i][j]'''
            NewTracker = [[ Tracker[i][j] if (i<i2 and i!=i1)   else   Tracker[i1][j]+Tracker[i2][j] if i==i1   else   Tracker[i+1][j]   for j in range(My)] for i in range(Mx-1)]
            Matrix = NewMatrix
            Tracker = NewTracker

        indeces = which_indeces_merge(Matrix)
    
    assert is_permuation_matrix(Matrix) # Then also Tracker is 'permutation-like' and I can only extract the one non-empty element from its rows (or columns)
    ClusterMap = [ [] for row in Tracker ]
    for i,row in enumerate(Tracker):
        for j,el in enumerate(row):
            if len(el)>0:
                for jj in range(len(row)):
                    if jj!=j:
                        assert len(row[jj])==0, 'ERROR: Tracker is not permutation-like'
                ClusterMap[i] = el
                
    return ClusterMap

def setup_pipeline(filestring, Properties, TimestepInterval, AnalyseEvery, sector = False):

    trajname   = '{:s}/Results/Traj_{:s}.xyz'.format(folder,filestring)
    trajsecname = '{:s}/Results/TrajSector_{:s}.xyz'.format(folder,filestring)
    bondsname  = '{:s}/Results/Bonds_{:s}.dat'.format(folder,filestring)
    bondssecname  = '{:s}/Results/BondsSector_new_{:s}.dat'.format(folder,filestring)
    configname = '{:s}/Input/Configurations/Config_{:s}.dat'.format(folder,filestring)
    par = getparams(filestring)
    nF, lF, nM, Lz = par['nF'], par['lF'], par['nM'], par['Lz']

    try:
        if sector: pipeline = import_file(trajsecname, multiple_frames = True, sort_particles = True)
        else: pipeline = import_file(trajname, multiple_frames = True, sort_particles = True)
    except:
        return None,0,0

    # Manual modifications of the imported data objects:
    def modify_pipeline_input(frame: int, data: DataCollection):
        data.particles_.particle_types_.type_by_id_(4).color = (0.8705882430076599, 0.9254902005195618, 1.0)
        data.particles_.particle_types_.type_by_id_(4).radius = 0.5
        data.particles_.particle_types_.type_by_id_(5).color = (0.0, 0.9768062829971313, 0.0)
        data.particles_.particle_types_.type_by_id_(5).radius = 0.5
        data.particles_.particle_types_.type_by_id_(6).color = (0.07759212702512741, 0.6699779033660889, 0.0)
        data.particles_.particle_types_.type_by_id_(6).radius = 0.5
        try: data.particles_.particle_types_.type_by_id_(7).color = (0.004852368962019682, 0.0960860624909401, 0.574990451335907)
        except KeyError: pass
        try:data.particles_.particle_types_.type_by_id_(7).radius = 0.5
        except KeyError: pass
        try: data.particles_.particle_types_.type_by_id_(8).color = (0.15294118225574493, 0.45098039507865906, 1.0)
        except KeyError: pass
        try: data.particles_.particle_types_.type_by_id_(8).radius = 0.5
        except KeyError: pass
    pipeline.modifiers.append(modify_pipeline_input)

    # Visual element initialization:
    data = pipeline.compute() # Evaluate new pipeline to gain access to visual elements associated with the imported data objects.
    data.particles.vis.radius = 1.0
    data.cell.vis.rendering_color = (0.3019607961177826, 0.5411764979362488, 0.7764706015586853)
    del data # Done accessing input DataCollection of pipeline.

    # Load bonds:
    modBonds = LoadTrajectoryModifier()
    pipeline.modifiers.append(modBonds)
    if sector: 
        try: modBonds.source.load(bondssecname, columns = ['BondID','Particle Identifiers.1', 'Particle Identifiers.2', 'Bond Type', 'Length', 'Energy.1'], multiple_frames = True)
        except:
            print("no new sector")
            return None,0,0
    else: modBonds.source.load(bondsname, columns = ['BondID','Particle Identifiers.1', 'Particle Identifiers.2', 'Bond Type', 'Length', 'Energy.1'], multiple_frames = True)


    #### Skip file if no new line to add to dicR
    NumberOfFrames = np.min([modBonds.source.num_frames, pipeline.source.num_frames])
    if not sector:
        if filestring in dicR.keys():
            dfOld = dicR[filestring]
            if len(dfOld)==int(NumberOfFrames/(AnalyseEvery/TimestepInterval)) and np.array([p in dfOld.columns for p in Properties]).all():
                print("{:s} - Nothing new".format(filestring))
                return None,0,0
            else:
                dfOld = {}
                PropertiesToCompute = Properties #[p for p in Properties if p not in dfOld]
        else:
            PropertiesToCompute = Properties
    else:
        if 'Sector_'+filestring in dicR.keys():
            dfOld = dicR['Sector_'+filestring]
            if len(dfOld)==int(NumberOfFrames/(AnalyseEvery/TimestepInterval)) and np.array([p in dfOld.columns for p in Properties]).all():
                print("{:s} - Nothing new".format(filestring))
                return None,0,0
            else:
                dfOld = {}
                PropertiesToCompute = Properties #[p for p in Properties if p not in dfOld]
        else:
            PropertiesToCompute = Properties

    if sector: print("Sector_{:s} - Analyse".format(filestring))
    else: print("{:s} - Analyse".format(filestring))
    print(PropertiesToCompute)

    #### Define OVITO Modifiers
    # Compute property Rcenter:
    pipeline.modifiers.append(ComputePropertyModifier(
        expressions = ('sqrt(Position.X^2 + Position.Y^2)',), # ((ParticleType>=4)*(ParticleType<=6))
        output_property = 'Rcenter'))

    # Compute molecule ID:
    pipeline.modifiers.append(ComputePropertyModifier(
        expressions = ('(ParticleIdentifier<={nF}*{lF})*rint(ParticleIdentifier/{lF}+0.49999) + (ParticleIdentifier>{nF}*{lF})*({nF}+rint((ParticleIdentifier-{nF}*{lF})/2+0.49999))'.format(nF=nF,lF=lF),),
        output_property = 'Molec'))

    # Compute PolyTime (proxy for filament orientation):
    pipeline.modifiers.append(ComputePropertyModifier(
        expressions = (' ((ParticleType>=4)*(ParticleType<=6)) * (ParticleIdentifier-{lF}*(Molec-1)) '.format(lF=lF),),
        output_property = 'PolyTime'))

    # Compute ThetaRad:
    pipeline.modifiers.append(ComputePropertyModifier(
        expressions = ('atan2(Position.Y,Position.X)',),
        output_property = 'ThetaRad'))

    if True:
        # Compute property PolytimeBoundActin for AMbond: polytime dell'atomo di actina in un AM bond
        pipeline.modifiers.append(ComputePropertyModifier(
            operate_on = 'bonds',
            expressions = ('(BondType==7) * ( @1.PolyTime*(@1.ParticleType==6) + @2.PolyTime*(@2.ParticleType==6) ) - 1000*(BondType!=7)',),
            output_property = 'PolytimeA_AMbond'))

        # Ale: Compute property PolytimeBoundActin for ACbond: polytime dell'atomo di actina in un AC bond
        pipeline.modifiers.append(ComputePropertyModifier(
            operate_on = 'bonds',
            expressions = ('(BondType==9) * ( @1.PolyTime*(@1.ParticleType==6) + @2.PolyTime*(@2.ParticleType==6) ) - 1000*(BondType!=9)',),
            output_property = 'PolytimeA_ACbond'))

        # Compute property MolecBoundActin for AMbond: id molecolare della actina di un AM bond
        pipeline.modifiers.append(ComputePropertyModifier(
            operate_on = 'bonds',
            expressions = ('(BondType==7) * ( @1.Molec*(@1.ParticleType==6) + @2.Molec*(@2.ParticleType==6) ) - 1000*(BondType!=7)',),
            output_property = 'MolecA_AMbond'))
        
        # Ale: Compute property MolecBoundActin for ACbond: id molecolare della actina di un AC bond
        pipeline.modifiers.append(ComputePropertyModifier(
            operate_on = 'bonds',
            expressions = ('(BondType==9) * ( @1.Molec*(@1.ParticleType==6) + @2.Molec*(@2.ParticleType==6) ) - 1000*(BondType!=9)',),
            output_property = 'MolecA_ACbond'))

        # Direction for AAbond (oriented from barbed to pointed end)
        # Direction - DirectionX:
        if par['geom']=='rw':
            pipeline.modifiers.append(ComputePropertyModifier(
            operate_on = 'bonds',
            expressions = ('(BondType==4) * sign(@2.PolyTime-@1.PolyTime)*(@1.Position.X-@2.Position.X)',
                            '(BondType==4) * sign(@2.PolyTime-@1.PolyTime)*(@1.Position.Y-@2.Position.Y)',
                            '(BondType==4) * sign(@2.PolyTime-@1.PolyTime)*(@1.Position.Z-@2.Position.Z)'),
            output_property = 'DirectionAA_AAbond'))
        else:
            pipeline.modifiers.append(ComputePropertyModifier(
                operate_on = 'bonds',
                expressions = ('(BondType==4) * sign(@2.PolyTime-@1.PolyTime)*(@1.Position.X-@2.Position.X)',
                                '(BondType==4) * sign(@2.PolyTime-@1.PolyTime)*(@1.Position.Y-@2.Position.Y)',
                                '(BondType==4) * sign(@2.PolyTime-@1.PolyTime)*(@1.Position.Z-@2.Position.Z-PeriodicImage.Z*CellSize.Z)'),
                output_property = 'DirectionAA_AAbond'))

        # User-defined modifier 'Python script for Direction: spread Direction from Act-Act bond to Act particle':
        def spread_fromAAbond_toApart(frame: int, data: DataCollection):
            if data.particles != None:
                bonds=data.particles.bonds
                data.particles_.create_property('DirectionAA_Apart',dtype=float, components=3)
                for i,bondtype in enumerate(bonds.bond_types[:]):
                    if bondtype==4:
                        j0,j1 = bonds.topology[i]
                        if data.particles['Particle Type'][j0]==5 or data.particles['Particle Type'][j1]==5:
                            IndexList = [j0,j1]
                        else:
                            j = j0 if data.particles['PolyTime'][j0] < data.particles['PolyTime'][j1] else j1
                            IndexList = [j]
                        for j in IndexList:
                            data.particles_['DirectionAA_Apart'][j] = bonds['DirectionAA_AAbond'][i]
        pipeline.modifiers.append(spread_fromAAbond_toApart)

        # Spread Direction from Apart to AMbond, if bound
        # Direction - DirectionX on Myo-Act bond:
        pipeline.modifiers.append(ComputePropertyModifier(
            operate_on = 'bonds',
            expressions = ('(BondType==7) * ( @1.DirectionAA_Apart.1*(@1.ParticleType==6) + @2.DirectionAA_Apart.1*(@2.ParticleType==6) ) - 1000*(BondType!=7)',
                            '(BondType==7) * ( @1.DirectionAA_Apart.2*(@1.ParticleType==6) + @2.DirectionAA_Apart.2*(@2.ParticleType==6) ) - 1000*(BondType!=7)',
                            '(BondType==7) * ( @1.DirectionAA_Apart.3*(@1.ParticleType==6) + @2.DirectionAA_Apart.3*(@2.ParticleType==6) ) - 1000*(BondType!=7)'),
            output_property = 'DirectionAA_AMbond'))
        
        #Ale: spread direction from Apart to ACbond, if bound
        pipeline.modifiers.append(ComputePropertyModifier(
            operate_on = 'bonds',
            expressions = ('(BondType==9) * ( @1.DirectionAA_Apart.1*(@1.ParticleType==6) + @2.DirectionAA_Apart.1*(@2.ParticleType==6) ) - 1000*(BondType!=9)',
                            '(BondType==9) * ( @1.DirectionAA_Apart.2*(@1.ParticleType==6) + @2.DirectionAA_Apart.2*(@2.ParticleType==6) ) - 1000*(BondType!=9)',
                            '(BondType==9) * ( @1.DirectionAA_Apart.3*(@1.ParticleType==6) + @2.DirectionAA_Apart.3*(@2.ParticleType==6) ) - 1000*(BondType!=9)'),
            output_property = 'DirectionAA_ACbond'))

        # User-defined modifier 'Python script for PolytimeOfActinBoundToThisMyo and DirectionOfActinBoundToThisMyo: spread them from AMbond to Mpart':
        def spread_fromAMbond_toMpart(frame: int, data: DataCollection):
            if data.particles != None:
                bonds=data.particles.bonds
                data.particles_.create_property('PolytimeA_Mpart',dtype=int, components=1)
                data.particles_.create_property('MolecA_Mpart',dtype=int, components=1)
                data.particles_.create_property('DirectionAA_Mpart',dtype=float, components=3)
                for i,bondtype in enumerate(bonds.bond_types[:]):
                    if bondtype==7:
                        topology = bonds.topology[i]
                        index = topology[0] if data.particles.particle_types[topology[0]]==8 else topology[1]
                        if data.particles.particle_types[index]!=8:
                            print("  Irregular bond: bond index {:}, part index {:d}-{:d}, part id {:d} has type {:d}. Check Bonds file if too many irregular bonds".format(
                                i, topology[0],topology[1], data.particles.identifiers[index], data.particles.particle_types[index]))
                        data.particles_['PolytimeA_Mpart'][index] = bonds['PolytimeA_AMbond'][i]
                        data.particles_['MolecA_Mpart'][index] = bonds['MolecA_AMbond'][i]
                        data.particles_['DirectionAA_Mpart'][index] = bonds['DirectionAA_AMbond'][i]
        pipeline.modifiers.append(spread_fromAMbond_toMpart)

        #Ale: spread from AMbond to Apart                                                                           !!!!!!!
        def spread_fromAMbond_toApart(frame: int, data: DataCollection):
            if data.particles != None:
                bonds=data.particles.bonds
                data.particles_.create_property('BondedM_Apart',dtype=int, components=1)
                for i,bondtype in enumerate(bonds.bond_types[:]):
                    if bondtype==7:
                        topology = bonds.topology[i]
                        index = topology[0] if data.particles.particle_types[topology[0]]==6 else topology[1]
                        if data.particles.particle_types[index]!=6:
                            print("  Irregular bond: bond index {:}, part index {:d}-{:d}, part id {:d} has type {:d}. Check Bonds file if too many irregular bonds".format(
                                i, topology[0],topology[1], data.particles.identifiers[index], data.particles.particle_types[index]))
                        data.particles_['BondedM_Apart'][index] = 1 
        pipeline.modifiers.append(spread_fromAMbond_toApart)

        #Ale: spread from ACbond to Apart                                                                       !!!!!!!
        def spread_fromACbond_toApart(frame: int, data: DataCollection):
            if data.particles != None:
                bonds=data.particles.bonds
                data.particles_.create_property('BondedC_Apart',dtype=int, components=1)
                for i , bondtype in enumerate(bonds.bond_types[:]):
                    if bondtype==9:
                        topology = bonds.topology[i]
                        index = topology[0] if data.particles.particle_types[topology[0]]==6 else topology[1] #id of the actin particle
                        if data.particles.particle_types[index]!=6:
                            print("  Irregular bond: bond index {:}, part index {:d}-{:d}, part id {:d} has type {:d}. Check Bonds file if too many irregular bonds".format(
                                i, topology[0],topology[1], data.particles.identifiers[index], data.particles.particle_types[index]))
                        data.particles_['BondedC_Apart'][index] = 1 
        pipeline.modifiers.append(spread_fromACbond_toApart)

        #Ale: spread from ACbond to Cpart 
        def spread_fromACbond_toCpart(frame: int, data: DataCollection):
            if data.particles != None:
                bonds=data.particles.bonds
                data.particles_.create_property('PolytimeA_Cpart',dtype=int, components=1)
                data.particles_.create_property('MolecA_Cpart',dtype=int, components=1)
                data.particles_.create_property('DirectionAA_Cpart',dtype=float, components=3)
                for i,bondtype in enumerate(bonds.bond_types[:]):
                    if bondtype==9:
                        topology = bonds.topology[i]
                        index = topology[0] if data.particles.particle_types[topology[0]]==10 else topology[1]
                        if data.particles.particle_types[index]!=10: #crosslinker bound is type 10
                            print("  Irregular bond: bond index {:}, part index {:d}-{:d}, part id {:d} has type {:d}. Check Bonds file if too many irregular bonds".format(
                                i, topology[0],topology[1], data.particles.identifiers[index], data.particles.particle_types[index]))
                        data.particles_['PolytimeA_Cpart'][index] = bonds['PolytimeA_ACbond'][i]
                        data.particles_['MolecA_Cpart'][index] = bonds['MolecA_ACbond'][i]
                        data.particles_['DirectionAA_Cpart'][index] = bonds['DirectionAA_ACbond'][i]
        pipeline.modifiers.append(spread_fromACbond_toCpart)


        # Mbonds - Expression selection:
        pipeline.modifiers.append(ExpressionSelectionModifier(
            expression = '(BondType==6)*(@1.ParticleType==8)*(@2.ParticleType==8)',
            operate_on = 'bonds'))
        # Mbonds - Parallel/Antiparallel (=1 if parallel, =-1 if antiparallel):
        pipeline.modifiers.append(ComputePropertyModifier(
            operate_on = 'bonds',
            expressions = ('sign(@1.DirectionAA_Mpart.1 * @2.DirectionAA_Mpart.1 + @1.DirectionAA_Mpart.2 * @2.DirectionAA_Mpart.2 + @1.DirectionAA_Mpart.3 * @2.DirectionAA_Mpart.3)',),
            output_property = 'Parallel_MMbond',
            only_selected = True))
        # Mbonds - Couple (first 2 digits are Polytime of @1, last 2 digits are Polytime of @2):
        pipeline.modifiers.append(ComputePropertyModifier(
            operate_on = 'bonds',
            expressions = ('Parallel_MMbond*(100*min(@1.PolytimeA_Mpart,@2.PolytimeA_Mpart) + max(@1.PolytimeA_Mpart,@2.PolytimeA_Mpart) )',),
            output_property = 'EncodedPolytimeA_MMbond',
            only_selected = True))
        # Mbonds - Extensivity (goes from -50, most contractile, to +50, most extended config. Not very meaningful for parallel) :
        pipeline.modifiers.append(ComputePropertyModifier(
            operate_on = 'bonds',
            expressions = ('(@1.PolytimeA_Mpart + @2.PolytimeA_Mpart - {lF}) * (Parallel_MMbond==-1)  +  (@1.PolytimeA_Mpart + @2.PolytimeA_Mpart - {lF} -1000) * (Parallel_MMbond==1)'.format(lF=lF),),
            output_property = 'Extensivity_MMbond',
            only_selected = True))
        # Mbonds - StaggerStuck (-1 = both Myo at barbed end, 0 = no constriction, >0 = potental constriction) :
        pipeline.modifiers.append(ComputePropertyModifier(
            operate_on = 'bonds',
            expressions = ('-2000 * (Parallel_MMbond!=1 && Parallel_MMbond!=-1)  +  ( ({lF}-@1.PolytimeA_Mpart)*(@2.PolytimeA_Mpart>={lF}-3)*(@1.PolytimeA_Mpart<{lF}-3) + ({lF}-@2.PolytimeA_Mpart)*(@1.PolytimeA_Mpart>={lF}-3)*(@2.PolytimeA_Mpart<{lF}-3) - 1*(@2.PolytimeA_Mpart>={lF}-3)*(@1.PolytimeA_Mpart>={lF}-3) ) * (Parallel_MMbond==1 || Parallel_MMbond==-1)'.format(lF=lF),),
            output_property = 'StaggerStuck_MMbond',
            only_selected = True))
        pipeline.modifiers.append(ClearSelectionModifier(
            operate_on='bonds'))
        
        # set values to nonsense for all non-MMbonds or MMbonds representing not fully bound myosins
        pipeline.modifiers.append(ExpressionSelectionModifier(
            expression = '(BondType==6)*(1-(@1.ParticleType==8)*(@2.ParticleType==8)) + (BondType!=6)',
            operate_on = 'bonds'))
        # Contacts - Parallel (-2000):
        pipeline.modifiers.append(ComputePropertyModifier(
            operate_on = 'bonds',
            expressions = ('-2000',),
            output_property = 'Parallel_MMbond',
            only_selected = True))
        # Contacts - Couple (first 2 digits are Polytime of @1, last 2 digits are Polytime of @2):
        pipeline.modifiers.append(ComputePropertyModifier(
            operate_on = 'bonds',
            expressions = ('(BondType==6)*((@1.ParticleType==8)*@1.PolytimeA_Mpart + (@2.ParticleType==8)*@2.PolytimeA_Mpart)',),
            output_property = 'EncodedPolytimeA_MMbond',
            only_selected = True))
        # Contacts - Extensivity (-2000):
        pipeline.modifiers.append(ComputePropertyModifier(
            operate_on = 'bonds',
            expressions = ('-2000',),
            output_property = 'Extensivity_MMbond',
            only_selected = True))
        # Contacts - StaggerStuck (-2000):
        pipeline.modifiers.append(ComputePropertyModifier(
            operate_on = 'bonds',
            expressions = ('-2000',),
            output_property = 'StaggerStuck_MMbond',
            only_selected = True))
        pipeline.modifiers.append(ClearSelectionModifier(
            operate_on='bonds'))

        # Ale: Cbonds - Expression selection:
        pipeline.modifiers.append(ExpressionSelectionModifier(
            expression = '(BondType==8)*(@1.ParticleType==10)*(@2.ParticleType==10)',
            operate_on = 'bonds'))
        # Mbonds - Parallel/Antiparallel (=1 if parallel, =-1 if antiparallel):
        pipeline.modifiers.append(ComputePropertyModifier(
            operate_on = 'bonds',
            expressions = ('sign(@1.DirectionAA_Cpart.1 * @2.DirectionAA_Cpart.1 + @1.DirectionAA_Cpart.2 * @2.DirectionAA_Cpart.2 + @1.DirectionAA_Cpart.3 * @2.DirectionAA_Cpart.3)',),
            output_property = 'Parallel_CCbond',
            only_selected = True))
        pipeline.modifiers.append(ClearSelectionModifier(
            operate_on='bonds'))
        
        # set values to nonsense for all non-MMbonds or MMbonds representing not fully bound myosins
        pipeline.modifiers.append(ExpressionSelectionModifier(
            expression = '(BondType==8)*(1-(@1.ParticleType==10)*(@2.ParticleType==10)) + (BondType!=8)',
            operate_on = 'bonds'))
        # Contacts - Parallel (-2000):
        pipeline.modifiers.append(ComputePropertyModifier(
            operate_on = 'bonds',
            expressions = ('-2000',),
            output_property = 'Parallel_CCbond',
            only_selected = True))
        pipeline.modifiers.append(ClearSelectionModifier(
            operate_on='bonds'))

        # User-defined modifier 'Python script to spread Extensivity and Parallel from Myo-Myo bond to Myo part':
        def spread_fromMMbond_toMpart(frame: int, data: DataCollection):
            if data.particles != None:
                bonds=data.particles.bonds
                data.particles_.create_property('Extensivity_Mpart',dtype=int, components=1)
                data.particles_.create_property('ParallelMM_Mpart',dtype=int, components=1)
                for i,bondtype in enumerate(bonds.bond_types[:]):
                    if bondtype==6:
                        topology = bonds.topology[i]
                        for index in topology:
                            assert data.particles.particle_types[index] in [7,8], "{:d}-{:d}  part {:d}  type {:d}".format(topology[0],topology[1], data.particles.identifiers[index], data.particles.particle_types[index])
                            data.particles_['Extensivity_Mpart'][index] = bonds['Extensivity_MMbond'][i]
                            data.particles_['ParallelMM_Mpart'][index] = bonds['Parallel_MMbond'][i]                                
        pipeline.modifiers.append(spread_fromMMbond_toMpart)

        # Ale: User-defined modifier 'Python script to spread Extensivity and Parallel from Myo-Myo bond to Myo part':
        def spread_fromCCbond_toCpart(frame: int, data: DataCollection):
            if data.particles != None:
                bonds=data.particles.bonds
                data.particles_.create_property('ParallelCC_Cpart',dtype=int, components=1)
                for i,bondtype in enumerate(bonds.bond_types[:]):
                    if bondtype==8:
                        topology = bonds.topology[i]
                        for index in topology:
                            assert data.particles.particle_types[index] in [9,10], "{:d}-{:d}  part {:d}  type {:d}".format(topology[0],topology[1], data.particles.identifiers[index], data.particles.particle_types[index])
                            data.particles_['ParallelCC_Cpart'][index] = bonds['Parallel_CCbond'][i]                                
        pipeline.modifiers.append(spread_fromCCbond_toCpart)

        
        # Strain of Myo bonds - Compute property:
        pipeline.modifiers.append(ComputePropertyModifier(
            operate_on = 'bonds',
            expressions = ('Length/{:.1f} * (BondType==6)*(@1.ParticleType==8)*(@2.ParticleType==8)*(Parallel_MMbond==-1)'.format(MMbondLength),),
            output_property = 'Strain'))

    # Cluster actin, first by cutoff (if depletion is on), then by bonds
    if par['eAA']>0:
        pipeline.modifiers.append(SelectTypeModifier(types = {4, 5, 6}))
        pipeline.modifiers.append(ClusterAnalysisModifier(
            cutoff = 1.5,
            sort_by_size = True,
            only_selected = True,
            cluster_coloring = False))
        pipeline.modifiers.append(ComputePropertyModifier(
            expressions = ('Cluster',), # ((ParticleType>=4)*(ParticleType<=6))
            output_property = 'ClusterCutoff'))
    pipeline.modifiers.append(SelectTypeModifier(types = {4, 5, 6, 7, 8, 9, 10}))
    pipeline.modifiers.append(ClusterAnalysisModifier(
        neighbor_mode = ClusterAnalysisModifier.NeighborMode.Bonding,
        sort_by_size = True,
        only_selected = True,
        cluster_coloring = False))
    pipeline.modifiers.append(ComputePropertyModifier(
        expressions = ('Cluster',), # ((ParticleType>=4)*(ParticleType<=6))
        output_property = 'ClusterBond'))
    pipeline.modifiers.append(ClearSelectionModifier())

    return pipeline, NumberOfFrames, PropertiesToCompute

def analyse_frames(par, pipeline, NumberOfFrames, PropertiesToCompute, TimestepInterval, AnalyseEvery, sector = False):
    # Initialise
    pipeline.compute(frame=0)
    prevtimestep=-1
    linesPerFrame = []
    linecolumns=[]
    RingIntactThisFrame = 1
    dfsPartEarlier = []
    dfsBondMMEarlier = []
    dfPartM_prevts = [] # see where used
    dfPartAB_prevts = []
    dfPartM_prts = [] # see where used
    dfPartMB_prts = []

    nF, lF, nM, Lz = par['nF'], par['lF'], par['nM'], par['Lz']

    # Loop over frames

    for ThisFrame in range(0,NumberOfFrames):
        """if (ThisFrame+1)%(AnalyseEvery/TimestepInterval) !=0:
            continue"""
        print("Frame attempted {:d}".format(ThisFrame))
        data=pipeline.compute(frame=ThisFrame)
        timestep = data.attributes['Timestep']
        print("Frame imported {:d} at timestep {:d}. I analyse every {:d}".format(ThisFrame,timestep, AnalyseEvery))
        if timestep%AnalyseEvery==0 and timestep!=prevtimestep:
            line = []
            linecolumns = []
            print("Frame {:d}, timestep {:d}".format(ThisFrame,timestep))
            line.append(timestep)
            linecolumns.append('time')
            try:
                dfOld_ts = dfOld[dfOld['time']==timestep].iloc[0]
            except:
                dfOld_ts = pd.DataFrame([],columns=[])

            # import particle data as dataframe
            columns = []
            dummy = []
            for i in data.particles.properties:
                '''if i.identifier=='Position':
                    pos=np.array(i[:])
                    columns.append('x')
                    dummy.append(pos[:,0])
                    columns.append('y')
                    dummy.append(pos[:,1])
                    columns.append('z')
                    dummy.append(pos[:,2])'''
                if i.identifier=='Color':
                    continue
                else:
                    columns.append(i.identifier.replace(" ", ""))
                    dummy.append(i[:])
            dummy = [[x[i] for x in dummy] for i in range(len(dummy[0]))]
            dfPart = pd.DataFrame(dummy,columns=columns)
            for c in ['ParticleIdentifier','ParticleType','Molec','PolyTime','PolytimeA_Mpart','MolecA_Mpart','BondedM_Apart','Extensivity_Mpart','ClusterCutoff','ClusterBond','Cluster']:
                try:
                    dfPart[c] = dfPart[c].astype(int)
                except:
                    pass
            dfPart['Jammed'] = 0
            dfPartA = dfPart[dfPart.ParticleType.isin([4,5,6])] # A = actin
            dfPartA_to_M = dfPartA[dfPartA['BondedM_Apart']==True]
            dfPartA_to_C = dfPartA[dfPartA['BondedC_Apart']==True]
            dfPartAP = dfPart[dfPart.ParticleType.isin([4])] # AP = actin pointed end
            dfPartAB = dfPart[dfPart.ParticleType.isin([5])] # AB = actin barbed end #ATTENZIONE: non è actin bound
            dfPartMU = dfPart[dfPart.ParticleType.isin([7])] # MU = myosin unbound
            dfPartMB = dfPart[dfPart.ParticleType.isin([8])] # MB = myosin bound
            dfPartM = dfPart[dfPart.ParticleType.isin([7,8])] # M = all myosin
            dfPartCU = dfPart[dfPart.ParticleType.isin([9])] # CU = crosslinkers unbound
            dfPartCB = dfPart[dfPart.ParticleType.isin([10])] # CB = crosslinkers bound
            dfPartC = dfPart[dfPart.ParticleType.isin([9,10])] # C = all crosslinkers

            # import bond data as dataframe
            columns = []
            dummy = []
            if data.particles.bonds.count==0:
                '''linesPerFrame.append([timestep, 0, 0, [], [], [],
                                    [], [], 0, 0, 1,
                                    [], [], [],
                                    [], [],
                                    [], [],
                                    ])
                continue'''
                dfBond = pd.DataFrame(np.array([[-1,-1,-1,-1]]),columns=['BondID','BondType','TopologyA','TopologyB'])
                pass
            else:
                for i in data.particles.bonds.properties:
                    if i.identifier=='Topology':
                        top=np.array(i[:])
                        top = np.array([[min(x),max(x)] for x in top])  # order Topology such that TopologyA<TopologyB
                        columns.append('TopologyA')
                        dummy.append(top[:,0])
                        columns.append('TopologyB')
                        dummy.append(top[:,1])
                    elif i.identifier=='Periodic Image':
                        continue
                    else:
                        columns.append(i.identifier.replace(" ", ""))
                        dummy.append(i[:])
                dummy = [[x[i] for x in dummy] for i in range(len(dummy[0]))]
                dfBond = pd.DataFrame(dummy,columns=columns)
            for c in ['BondID','BondType','TopologyA','TopologyB','PolytimeA_AMbond','MolecA_AMbond','PolytimeA_ACbond','MolecA_ACbond','Parallel_MMbond','Parallel_CCbond','EncodedPolytimeA_MMbond','Extensivity_MMbond']:
                try:
                    dfBond[c] = dfBond[c].astype(int)
                except:
                    pass
            dfBondMM = dfBond[dfBond.BondType.isin([6])]
            dfBondAM = dfBond[dfBond.BondType.isin([7])]
            dfBondAA = dfBond[dfBond.BondType.isin([4])]
            dfBondAC = dfBond[dfBond.BondType.isin([9])]

            # merge ClusterBond with ClusterCutoff:
            if len(dfPartA)>0:
                if 'ClusterCutoff' in dfPartA and 'ClusterBond' in dfPartA:
                    """tic = []
                    ticdescription = []
                    tic.append(time.time())"""
                    Clusterdic = {}
                    ClusterCutoffList = dfPartA['ClusterCutoff'].unique()
                    ClusterCutoffList.sort()
                    NClusterCutoff = len(ClusterCutoffList)
                    assert NClusterCutoff==ClusterCutoffList.max(), "ClusterCutoffList ERROR"
                    ClusterCutoffParts = [[] for i in range(NClusterCutoff)] # Notice that entry i refers to cluster i+i
                    ClusterCutoffSizes = np.zeros(NClusterCutoff) # Notice that entry i refers to cluster i+i

                    ClusterBondList = dfPartA['ClusterBond'].unique()
                    ClusterBondList.sort()
                    NClusterBond = len(ClusterBondList)
                    assert NClusterBond==ClusterBondList.max(), "ClusterBondList ERROR"
                    #ClusterBondParts = [[] for i in range(NClusterBond)] # Notice that entry i refers to cluster i+i
                    #ClusterBondSizes = np.zeros(NClusterBond) # Notice that entry i refers to cluster i+i

                    """tic.append(time.time())
                    ticdescription.append('Init')"""
                    # Build adjacency matrix and create list of particles belonging to a given clusters
                    AdjacencyMatrix = np.zeros((NClusterCutoff,NClusterBond), dtype=int)  # Notice that entry i, j refers to clusters i+i and j+1 !
                    for cc in ClusterCutoffList:
                        cc_=cc-1 # 0-based index
                        dfThisClusterCutoffParts = dfPartA[dfPartA['ClusterCutoff']==cc]
                        ClusterCutoffParts[cc_] = dfThisClusterCutoffParts.index.values.tolist()
                        ClusterCutoffSizes[cc_]=len(ClusterCutoffParts[cc_])
                        for cb in ClusterBondList:
                            cb_ = cb-1 # 0-based index
                            #ClusterBondParts[cb_]=dfPartA[dfPartA['ClusterBond']==cb]['ParticleIdentifier'].values
                            #ClusterBondSizes[cb_]=len(ClusterBondParts[cb_])
                            if len(dfThisClusterCutoffParts[ dfThisClusterCutoffParts['ClusterBond']==cb ])>0:
                                AdjacencyMatrix[cc_,cb_]=1
                    assert ClusterCutoffSizes.sum()==len(dfPartA)
                    """tic.append(time.time())
                    ticdescription.append('AdjMat')"""

                    # Merge clusters from adjacency matrix and sort them according to new size
                    ClusterMap = mergeclusters(AdjacencyMatrix)
                    """tic.append(time.time())
                    ticdescription.append('mergeclusters')"""

                    Sizes = np.zeros(len(ClusterMap), dtype=int)
                    for c_,row in enumerate(ClusterMap):
                        cclist = list(set([ij[0] for ij in row])) # take all unique cutoffclusters in cluster c_
                        for cc_ in cclist:
                            Sizes[c_] += ClusterCutoffSizes[cc_]
                    """tic.append(time.time())
                    ticdescription.append('NewSizes')"""

                    ClusterMap = [c for s,c in sorted(zip(Sizes,ClusterMap), key=lambda pair: -pair[0])]
                    Sizes = [s for s,c in sorted(zip(Sizes,ClusterMap), key=lambda pair: -pair[0])]
                    assert sum(Sizes)==len(dfPartA)
                    """tic.append(time.time())
                    ticdescription.append('Sort')"""

                    dfClusterColumn = []
                    indexes = []
                    for c_,row in enumerate(ClusterMap):
                        cclist = list(set([ij[0] for ij in row]))
                        for cc_ in cclist:
                            newindexes = ClusterCutoffParts[cc_]
                            indexes += newindexes
                            dfClusterColumn += [c_+1 for i in newindexes]
                    dfClusterColumn = pd.Series(dfClusterColumn)
                    dfClusterColumn.index = indexes
                    dfPartA = dfPartA.copy()
                    dfPartA['Cluster'] = dfClusterColumn
                    print(" Reduced {:d} cutoff-clusters and {:d} bond-clusters to {:d} clusters".format(len(ClusterCutoffList),len(ClusterBondList),len(ClusterMap)))
                    """tic.append(time.time())
                    ticdescription.append('NewDf')"""

                    """tic = [tic[i+1]-tic[i] for i in range(len(tic)-1) ]
                    ticdescription = ['{:s}: {:.5f}'.format(ticdescription[i], tic[i]) for i in range(len(tic))]
                    print(ticdescription)"""
                elif 'ClusterBond' in dfPartA:
                    dfPartA = dfPartA.copy()
                    dfPartA['Cluster']= dfPartA['ClusterBond']
                elif 'ClusterCutoff' in dfPartA:
                    dfPartA = dfPartA.copy()
                    dfPartA['Cluster']= dfPartA['ClusterCutoff']


            #######################################
            # Start analysis of PropertiesToCompute

            # check whether ring is intact:
            if RingIntactThisFrame==1:
                if len(dfPartA.Cluster.unique())>1:
                    if len(dfPartA[dfPartA['Cluster']==1]) < 0.9*par['lF']*par['nF']:
                        print(" RING BROKEN")
                        RingIntactThisFrame = 0
                        #pipeline.add_to_scene()
                        #img = vp.render_image(size=(800,600), background=(1,1,1), filename='{:s}/Analysis/Screenshots/RingBreak_{:s}.png'.format(folder,filestring), frame=ThisFrame)
                if len(dfPartA)==0:
                    print(" RING BROKEN")
                    RingIntactThisFrame = 0
            #else:
            #    if len(dfPartA[dfPartA['Cluster']==1]) < 0.9*par['lF']*par['nF']:
            #        RingIntactThisFrame = 0
            line.append(RingIntactThisFrame)
            linecolumns.append('RingIntactFlag')
            
            # check actin percolated fraction:              !
            if 'PercolatedFraction' in PropertiesToCompute:
                if len(dfPartA)==0:
                    PercolatedFraction=0
                else:
                    PercolatedFraction = len(dfPartA[dfPartA['Cluster']==1])/len(dfPartA)
                print(" Percolated fraction is {:f}".format(PercolatedFraction))
            else:
                PercolatedFraction = -1
            line.append(PercolatedFraction)
            linecolumns.append('PercolatedFraction')

            # distance of actin particles from centre                   !
            RavgThisFrame = dfPartA.Rcenter.mean()
            RstdThisFrame = dfPartA.Rcenter.std()
            print(" Ravg is {:f} ±{:f}".format(RavgThisFrame,RstdThisFrame))
            line.append(RavgThisFrame)
            linecolumns.append('Ravg')
            line.append(RstdThisFrame)
            linecolumns.append('Rstd')

            # number of actin particles
            NActin = len(dfPartA)
            line.append(NActin)
            linecolumns.append('NActin')

            # histogram of actin density
            hist, bins = np.histogram(dfPartA['ThetaRad'], bins=24, range=[-np.pi,np.pi], density=False)
            #binsA = [ 0.5*(binsA[i]+binsA[i+1]) for i in range(0,len(binsA)-1)]
            histAdensityThisFrame = [bins, hist]
            line.append(histAdensityThisFrame)
            linecolumns.append('histAdensity')

            # histogram of actin barbed end density
            hist, bins = np.histogram(dfPartAB['ThetaRad'], bins=24, range=[-np.pi,np.pi], density=False)
            histABdensityThisFrame = [bins, hist]
            line.append(histABdensityThisFrame)
            linecolumns.append('histABdensity')

            # histogram of actin pointed end density
            hist, bins = np.histogram(dfPartAP['ThetaRad'], bins=24, range=[-np.pi,np.pi], density=False)
            histAPdensityThisFrame = [bins, hist]
            line.append(histAPdensityThisFrame)
            linecolumns.append('histAPdensity')

            # histogram of unbound myosin density
            hist, bins = np.histogram(dfPartMU['ThetaRad'], bins=24, range=[-np.pi,np.pi], density=False)
            #binsM = [ 0.5*(binsM[i]+binsM[i+1]) for i in range(0,len(binsM)-1)]
            histMUdensityThisFrame = [bins, hist]
            line.append(histMUdensityThisFrame)
            linecolumns.append('histMUdensity')

            # histogram of bound myosin density
            hist, bins = np.histogram(dfPartMB['ThetaRad'], bins=24, range=[-np.pi,np.pi], density=False)
            #binsM = [ 0.5*(binsM[i]+binsM[i+1]) for i in range(0,len(binsM)-1)]
            histMBdensityThisFrame = [bins, hist]
            line.append(histMBdensityThisFrame)
            linecolumns.append('histMBdensity')

            # Ale: histogram of unbound crosslinker density
            hist, bins = np.histogram(dfPartCU['ThetaRad'], bins=24, range=[-np.pi,np.pi], density=False)
            #binsM = [ 0.5*(binsM[i]+binsM[i+1]) for i in range(0,len(binsM)-1)]
            histCUdensityThisFrame = [bins, hist]
            line.append(histCUdensityThisFrame)
            linecolumns.append('histCUdensity')

            # Ale: histogram of bound crosslinker density
            hist, bins = np.histogram(dfPartCB['ThetaRad'], bins=24, range=[-np.pi,np.pi], density=False)
            #binsM = [ 0.5*(binsM[i]+binsM[i+1]) for i in range(0,len(binsM)-1)]
            histCBdensityThisFrame = [bins, hist]
            line.append(histCBdensityThisFrame)
            linecolumns.append('histCBdensity')

            # actin strain distribution and average
            if np.array([x in PropertiesToCompute for x in ['ActinStrainAvg']]).any():
                hist, bins = np.histogram(dfBondAA['Length'], bins=40, range=[0.80,1.2], density=False)
                ActinStrainAvg = dfBondAA['Length'].mean()
                histActinStrainDistr = [bins, hist]
                line.append(histActinStrainDistr)
                linecolumns.append('histActinStrainDistr')
                line.append(ActinStrainAvg)
                linecolumns.append('ActinStrainAvg')

            # velocity dot direction for actin filaments
            if np.array([x in PropertiesToCompute for x in ['vdotnActinAvg']]).any() and not sector:
                if len(dfPartAB_prevts)==0:
                    dfPartAB_prevts = dfPartAB
                    timestep_prevts = timestep
                assert (dfPartAB['ParticleIdentifier'].values ==dfPartAB_prevts['ParticleIdentifier'].values).all()
                dfPartAB = dfPartAB.copy()
                if timestep!=timestep_prevts:
                    dfPartAB['DisplacementOverTime'] = (dfPartAB['Position']-dfPartAB_prevts['Position'])/(timestep-timestep_prevts)    
                else:
                    dfPartAB['DisplacementOverTime'] = (dfPartAB['Position']-dfPartAB_prevts['Position'])/AnalyseEvery 
                dfPartAB['vdotn'] = [np.dot(x,y)/np.linalg.norm(y) for x, y  in zip(dfPartAB['DisplacementOverTime'],dfPartAB['DirectionAA_Apart'])]
                #histrange = np.array([0.80,1.12])
                #histrange = histrange/AnalyseEvery  if timestep==timestep_prevts  else  histrange/(timestep-timestep_prevts)
                hist, bins = np.histogram(dfPartAB['vdotn'], bins=40, density=False)
                vdotnActinAvg = dfPartAB['vdotn'].mean()
                histvdotnActinDistr = [bins, hist]
                line.append(histvdotnActinDistr)
                linecolumns.append('histvdotnActinDistr')
                line.append(vdotnActinAvg)
                linecolumns.append('vdotnActinAvg')
                
                '''dfPartAB['AngularDisplacementOverTime'] = [np.arctan2(w[1],w[0]) for w in dfPartAB['DisplacementOverTime']/dfPartAB['Rcenter']]
                hist, bins = np.histogram(dfPartAB['DisplacementOverTime'], bins=40, density=False)
                vActinAvg = dfPartAB['DisplacementOverTime'].mean()
                histvActinDistr = [bins, hist]
                line.append(histvActinDistr)
                linecolumns.append('histvActinDistr')
                line.append(vActinAvg)
                linecolumns.append('vActinAvg')'''
                
            # histogram of PolyTime of actin bound to myosin                
            hist, bins = np.histogram(dfPartMB['PolytimeA_Mpart'], bins=lF, range=[0.5,lF+0.5], density=False)
            histMDistrAlongActin = [bins, hist]
            line.append(histMDistrAlongActin)
            linecolumns.append('histMDistrAlongActin')
            
            hist, bins = np.histogram(dfPartMB['PolytimeA_Mpart'][dfPartMB.ParallelMM_Mpart.isin([-1,1])], bins=lF, range=[0.5,lF+0.5], density=False)
            histDoubleBoundMDistrAlongActin = [bins, hist]
            line.append(histDoubleBoundMDistrAlongActin)
            linecolumns.append('histDoubleBoundMDistrAlongActin')

            # Ale: histogram of PolyTime of actin bound to crosslinker                
            hist, bins = np.histogram(dfPartCB['PolytimeA_Cpart'], bins=lF, range=[0.5,lF+0.5], density=False)
            histCDistrAlongActin = [bins, hist]
            line.append(histCDistrAlongActin)
            linecolumns.append('histCDistrAlongActin')
            
            hist, bins = np.histogram(dfPartCB['PolytimeA_Cpart'][dfPartCB.ParallelCC_Cpart.isin([-1,1])], bins=lF, range=[0.5,lF+0.5], density=False)
            histDoubleBoundCDistrAlongActin = [bins, hist]
            line.append(histDoubleBoundCDistrAlongActin)
            linecolumns.append('histDoubleBoundCDistrAlongActin')
        
            #Ale: number of jammed particles
            if 'JammedMyosins' in PropertiesToCompute  or 'JammedActins' in PropertiesToCompute:
                JammedMyosins = 0
                try: 
                    JammedMyosins = dfPartMB['PolytimeA_Mpart'].value_counts()[lF-1]
                except KeyError: print('none at barbed end')
                print(f'At barbed end: {JammedMyosins}')

                line.append(JammedMyosins)
                linecolumns.append('JammedActins')
                
                jammed = []
                for id, row in dfPart.iterrows():
                    if row.ParticleType != 8: 
                        jammed.append(False)
                        continue
                    jammed.append(row['PolytimeA_Mpart'] == lF -1 or (row['PolytimeA_Mpart']+lF*(row['MolecA_Mpart']-1) in dfPartA_to_M['ParticleIdentifier'] and row['PolytimeA_Mpart']+lF*(row['MolecA_Mpart']-1)+1 % lF != 0))
                dfPart['Jammed'] = jammed
                try: JammedMyosins = dfPart['Jammed'].value_counts()[True]
                except KeyError: pass
                line.append(JammedMyosins)
                linecolumns.append('JammedMyosins')

                JammedDoubleBoundMyosins = 0
                try: JammedDoubleBoundMyosins = dfPart[(dfPart.ParallelMM_Mpart.isin([-1,1]))*(dfPart.ParticleType.isin([8]))==True]['Jammed'].value_counts()[True]
                except KeyError: pass
                line.append(JammedDoubleBoundMyosins)
                linecolumns.append('JammedDoubleBoundMyosins')

                print(f'Jammed: {JammedMyosins}')
                print(f'JammedDoubleBound: {JammedDoubleBoundMyosins}')

                line.append(len(dfPartMB[dfPartMB.ParallelMM_Mpart.isin([-1,1])])-JammedDoubleBoundMyosins)
                linecolumns.append('NotjammedDoubleBoundMyosins')

            #Ale: To have the jammed property also in the smaller dfs
            dfPartMU = dfPart[dfPart.ParticleType.isin([7])] # MU = myosin unbound
            dfPartMB = dfPart[dfPart.ParticleType.isin([8])] # MB = myosin bound
            dfPartM = dfPart[dfPart.ParticleType.isin([7,8])] # M = all myosin

            # Ale: number of other particles
            line.append(len(dfPartMB))
            linecolumns.append('BoundMyosin')
            line.append(len(dfPartMB[dfPartMB.ParallelMM_Mpart.isin([-1,1])]))
            linecolumns.append('DoubleboundMyosin')
            line.append(len(dfPartCB))
            linecolumns.append('BoundCrosslinkers')
            line.append(len(dfPartCB[dfPartCB.ParallelCC_Cpart.isin([-1,1])]))
            linecolumns.append('DoubleboundCrosslinkers')

            #Ale: end to end distance
            if 'EndtoendDistance' in PropertiesToCompute and not sector:
                EndtoendDistances = []
                for i in range(1,getparams(filestring)['nF']+1):
                    if len(dfPartAB[dfPartAB['Molec']==i])>0 and len(dfPartAP[dfPartAP['Molec']==i])>0:
                        EndtoendDistances.append(np.sqrt((dfPartAB[dfPartAB['Molec']==i]['Position'].values[0][0]-dfPartAP[dfPartAP['Molec']==i]['Position'].values[0][0])**2
                                                    +(dfPartAB[dfPartAB['Molec']==i]['Position'].values[0][1]-dfPartAP[dfPartAP['Molec']==i]['Position'].values[0][1])**2
                                                    +(dfPartAB[dfPartAB['Molec']==i]['Position'].values[0][2]-dfPartAP[dfPartAP['Molec']==i]['Position'].values[0][2])**2))
                    else: EndtoendDistances.append(np.nan)
                EndtoendDistances = np.array(EndtoendDistances)
                #print(EndtoendDistances)
                EndtoendDistance = np.nanmean(EndtoendDistances)
                print('Endtoend: '+str(EndtoendDistance))
                line.append(EndtoendDistance)
                linecolumns.append('EndtoendDistance')

                #End to end of filaments bonded to at least one not jammed M and one C
                mask = np.array([((1-dfPartMB[dfPartMB['MolecA_Mpart']==i]['Jammed']).any() and dfPartA[dfPartA['Molec']==i]['BondedC_Apart'].any())  for i in range(1,getparams(filestring)['nF']+1)])
                try: 
                    EndtoendDistanceBonded = np.nanmean(EndtoendDistances[mask])
                except:
                    EndtoendDistanceBonded = np.nan
                print('Endtoend Bonded MC ( '+str(np.sum(mask))+' / ' + str(getparams(filestring)['nF']) +' ): '+str(EndtoendDistanceBonded))
                line.append(EndtoendDistanceBonded)
                linecolumns.append('EndtoendDistanceBondedMC')

                try: 
                    EndtoendDistanceNotbonded = np.nanmean(EndtoendDistances[np.invert(mask)])
                except:
                    EndtoendDistanceNotBonded = np.nan
                print('Endtoend Not bonded MC ( '+str(np.sum(1-mask))+' / ' + str(getparams(filestring)['nF']) +' ): '+str(EndtoendDistanceNotbonded))
                line.append(EndtoendDistanceNotbonded)
                linecolumns.append('EndtoendDistanceNotbondedMC')

                #End to end of filaments bonded to at least one jammed myosin and one not jammed myosin
                mask = np.array([(dfPartMB[dfPartMB['MolecA_Mpart']==i]['Jammed'].any() and (1-dfPartMB[dfPartMB['MolecA_Mpart']==i]['Jammed']).any()) for i in range(1,getparams(filestring)['nF']+1)])
                try: 
                    EndtoendDistanceBonded = np.nanmean(EndtoendDistances[mask])
                except:
                    EndtoendDistanceBonded = np.nan
                print('Endtoend Bonded M jammed M not jammed ( '+str(np.sum(mask))+' / ' + str(getparams(filestring)['nF']) +' ): '+str(EndtoendDistanceBonded))
                line.append(EndtoendDistanceBonded)
                linecolumns.append('EndtoendDistanceBondedMM')
                try: 
                    EndtoendDistanceNotbonded = np.nanmean(EndtoendDistances[np.invert(mask)])
                except:
                    EndtoendDistanceNotBonded = np.nan
                print('Endtoend Not bonded M jammed M not jammed ( '+str(np.sum(1-mask))+' / ' + str(getparams(filestring)['nF']) +' ): '+str(EndtoendDistanceNotbonded))
                line.append(EndtoendDistanceNotbonded)
                linecolumns.append('EndtoendDistanceNotbondedMM')

                # End to end distance of filaments in ring and out of ring
                mask = np.array([(dfPartA[dfPartA['Molec']==i]['Cluster']==1).any() for i in range(1,getparams(filestring)['nF']+1)])
                try: 
                    EndtoendDistanceBonded = np.nanmean(EndtoendDistances[mask])
                except:
                    EndtoendDistanceBonded = np.nan
                print('Endtoend Bonded to smth ( '+str(np.sum(mask))+' / ' + str(getparams(filestring)['nF']) +' ): '+str(EndtoendDistanceBonded))
                line.append(EndtoendDistanceBonded)
                linecolumns.append('EndtoendDistanceBonded')
                try: 
                    EndtoendDistanceFree = np.nanmean(EndtoendDistances[np.invert(mask)])
                except:
                    EndtoendDistanceFree = np.nan
                print('Endtoend Free ( '+str(np.sum(np.invert(mask)))+' / ' + str(getparams(filestring)['nF']) +' ): '+str(EndtoendDistanceFree))
                line.append(EndtoendDistanceFree)
                linecolumns.append('EndtoendDistanceFree')

            #Ale: myosin strain distribution and average
            if np.array([x in PropertiesToCompute for x in ['DBNJMyosinStrainAvg','MyosinStrainAvg']]).any():
                print("i am computing myosin strain avg")
                df_temp = dfBondMM
                hist, bins = np.histogram(df_temp['Length'], bins=40, density=False)
                MyosinStrainAvg = df_temp['Length'].mean()
                print("Myosin Strain Avg is "+str(MyosinStrainAvg))
                histMyosinStrainDistr = [bins, hist]
                line.append(histMyosinStrainDistr)
                linecolumns.append('histMyosinStrainDistr')
                line.append(MyosinStrainAvg)
                linecolumns.append('MyosinStrainAvg')
                #New stuff:
                DoubleBoundOnejammedParallelMMbondlengths = []
                DoubleBoundNeitherjammedParallelMMbondlengths = []
                DoubleBoundOnejammedAntiparallelMMbondlengths = []
                DoubleBoundNeitherjammedAntiparallelMMbondlengths = []
                for id, bond in dfBondMM.iterrows():
                    if (dfPartM.loc[bond['TopologyA']]['Jammed']!=True and dfPartM.loc[bond['TopologyB']]['Jammed']!=True and dfPartM.loc[bond['TopologyA']]['ParticleType']==8 and dfPartM.loc[bond['TopologyB']]['ParticleType']==8): #if both bonded, both not jammed
                        if bond['Parallel_MMbond'] == 1:
                            DoubleBoundNeitherjammedParallelMMbondlengths.append(bond['Length'])
                        elif bond['Parallel_MMbond'] == -1:
                            DoubleBoundNeitherjammedAntiparallelMMbondlengths.append(bond['Length'])
                    elif (dfPartM.loc[bond['TopologyA']]['Jammed']!=True or dfPartM.loc[bond['TopologyB']]['Jammed']!=True) and dfPartM.loc[bond['TopologyA']]['ParticleType']==8 and dfPartM.loc[bond['TopologyB']]['ParticleType']==8:
                        if bond['Parallel_MMbond'] == 1:
                            DoubleBoundOnejammedParallelMMbondlengths.append(bond['Length'])
                        elif bond['Parallel_MMbond'] == -1: 
                            DoubleBoundOnejammedAntiparallelMMbondlengths.append(bond['Length'])
                try:
                    V = DoubleBoundOnejammedParallelMMbondlengths+DoubleBoundNeitherjammedParallelMMbondlengths+DoubleBoundOnejammedAntiparallelMMbondlengths+DoubleBoundNeitherjammedAntiparallelMMbondlengths
                    hist, bins = np.histogram(V,bins = 40, density = False)
                    MyosinStrainAvg = np.nanmean(V)
                    histDoubleBoundMyosinStrainDistr = [hist,bins]
                    line.append(histDoubleBoundMyosinStrainDistr)
                    linecolumns.append('histDoubleBoundMyosinStrainDistr')
                    line.append(MyosinStrainAvg)
                    linecolumns.append('DoubleBoundMyosinStrainAvg')
                    line.append(len(V))
                    linecolumns.append('DoubleBoundMM')
                    print("Double bonded Myosin Strain Avg is "+str(MyosinStrainAvg)+' with '+str(len(V))+' MM bonds')
                except:
                    print("couldn't compute double bonded myosin strain")
                    pass
                try:
                    hist, bins = np.histogram(DoubleBoundOnejammedParallelMMbondlengths,bins = 40, density = False)
                    MyosinStrainAvg = np.nanmean(DoubleBoundOnejammedParallelMMbondlengths)
                    histogram = [hist,bins]
                    line.append(histogram)
                    linecolumns.append('histDoubleBoundOnejammedParallelMyosinStrainDistr')
                    line.append(MyosinStrainAvg)
                    linecolumns.append('DoubleBoundOnejammedParallelMyosinStrainAvg')
                    line.append(len(DoubleBoundOnejammedParallelMMbondlengths))
                    linecolumns.append('DoubleBoundOnejammedParallelMM')
                    print("Double bonded one jammed parallel Myosin Strain Avg is "+str(MyosinStrainAvg)+' with '+str(len(DoubleBoundOnejammedParallelMMbondlengths))+' MM bonds')
                except:
                    print("couldn't compute double bonded one jammed parallel myosin strain")
                    pass
                try:
                    hist, bins = np.histogram(DoubleBoundNeitherjammedParallelMMbondlengths,bins = 40, density = False)
                    MyosinStrainAvg = np.nanmean(DoubleBoundNeitherjammedParallelMMbondlengths)
                    histogram = [hist,bins]
                    line.append(histogram)
                    linecolumns.append('histDoubleBoundNeitherjammedParallelMyosinStrainDistr')
                    line.append(MyosinStrainAvg)
                    linecolumns.append('DoubleBoundNeitherjammedParallelMyosinStrainAvg')
                    line.append(len(DoubleBoundNeitherjammedParallelMMbondlengths))
                    linecolumns.append('DoubleBoundNeitherjammedParallelMM')
                    print("Double bonded neither jammed parallel Myosin Strain Avg is "+str(MyosinStrainAvg)+' with '+str(len(DoubleBoundNeitherjammedParallelMMbondlengths))+' MM bonds')
                except:
                    print("couldn't compute double bonded neither jammed parallel myosin strain")
                    pass
                try:
                    hist, bins = np.histogram(DoubleBoundOnejammedAntiparallelMMbondlengths,bins = 40, density = False)
                    MyosinStrainAvg = np.nanmean(DoubleBoundOnejammedAntiparallelMMbondlengths)
                    histogram = [hist,bins]
                    line.append(histogram)
                    linecolumns.append('histDoubleBoundOnejammedAntiparallelMyosinStrainDistr')
                    line.append(MyosinStrainAvg)
                    linecolumns.append('DoubleBoundOnejammedAntiparallelMyosinStrainAvg')
                    line.append(len(DoubleBoundOnejammedAntiparallelMMbondlengths))
                    linecolumns.append('DoubleBoundOnejammedAntiparallelMM')
                    print("Double bonded one jammed antiparallel Myosin Strain Avg is "+str(MyosinStrainAvg)+' with '+str(len(DoubleBoundOnejammedAntiparallelMMbondlengths))+' MM bonds')
                except:
                    print("couldn't compute double bonded one jammed antiparallel myosin strain")
                    pass
                try:
                    hist, bins = np.histogram(DoubleBoundNeitherjammedAntiparallelMMbondlengths,bins = 40, density = False)
                    MyosinStrainAvg = np.nanmean(DoubleBoundNeitherjammedAntiparallelMMbondlengths)
                    histogram = [hist,bins]
                    line.append(histogram)
                    linecolumns.append('histDoubleBoundNeitherjammedAntiparallelMyosinStrainDistr')
                    line.append(MyosinStrainAvg)
                    linecolumns.append('DoubleBoundNeitherjammedAntiparallelMyosinStrainAvg')
                    line.append(len(DoubleBoundNeitherjammedAntiparallelMMbondlengths))
                    linecolumns.append('DoubleBoundNeitherjammedAntiparallelMM')
                    print("Double bonded neither jammed antiparallel Myosin Strain Avg is "+str(MyosinStrainAvg)+' with '+str(len(DoubleBoundNeitherjammedAntiparallelMMbondlengths))+' MM bonds')
                except:
                    print("couldn't compute double bonded neither jammed antiparallel myosin strain")
                    pass
                #Now double bound, not jammed myosins only
                """lens_dbnj = []
                for id, bond in dfBondMM.iterrows():
                    if (dfPartM.loc[bond['TopologyA']]['Jammed']!=True and dfPartM.loc[bond['TopologyA']]['ParticleType']==8 and dfPartM.loc[bond['TopologyB']]['ParticleType']==8): #if both bonded, first one not jammed
                        lens_dbnj.append(bond['Length'])
                    if (dfPartM.loc[bond['TopologyB']]['Jammed']!=True and dfPartM.loc[bond['TopologyA']]['ParticleType']==8 and dfPartM.loc[bond['TopologyB']]['ParticleType']==8): #if both bonded, first one not jammed
                        lens_dbnj.append(bond['Length'])
                hist, bins = np.histogram(lens_dbnj,bins = 40, density = False)
                MyosinStrainAvg = np.mean(lens_dbnj)
                print("Double bonded not jammed Myosin Strain Avg is "+str(MyosinStrainAvg))
                histMyosinStrainDistr = [bins, hist]
                line.append(histMyosinStrainDistr)
                linecolumns.append('histDBNJMyosinStrainDistr')
                line.append(MyosinStrainAvg)
                linecolumns.append('DBNJMyosinStrainAvg')"""

            # myosin speed (Ale modified!)
            if np.array([x in PropertiesToCompute for x in ['MSpeedParallelAvg']]).any():
                dfBondMM_ts = dfBondMM.sort_values(by='TopologyA').set_index('TopologyA',drop=False)
                for c in ['Length','Energy','PolytimeA_AMbond','MolecA_AMbond']:
                    del dfBondMM_ts[c]
                if len(dfPartM_prts)==0:
                    dfPartM_prts=dfPartM
                    dfBondMM_prts=dfBondMM_ts
                    timestep_prts=timestep

                if sector:
                    MSpeed = []
                    StillBoundAntiparallel = []
                    StillBoundAntiparallelNotjammed = []
                    StillBoundParallel = []
                    StillBoundParallelNotjammed = []
                    StillBoundParallelNotjammedTojammed = []
                    for _,b in dfBondMM_ts.iterrows():
                        if (b['TopologyA'],b['TopologyB']) in zip(dfBondMM_prts['TopologyA'],dfBondMM_prts['TopologyB']):
                            try: StillBoundAntiparallel.append(((dfPartM.loc[b.TopologyA]['MolecA_Mpart']==dfPartM_prts.loc[b.TopologyA]['MolecA_Mpart']) * (dfPartM.loc[b.TopologyA]['ParticleType']==8) * (dfPartM_prts.loc[b.TopologyA]['ParticleType']==8)) * (
                                                (dfPartM.loc[b.TopologyB]['MolecA_Mpart']==dfPartM_prts.loc[b.TopologyB]['MolecA_Mpart']) * (dfPartM.loc[b.TopologyB]['ParticleType']==8) * (dfPartM_prts.loc[b.TopologyB]['ParticleType']==8)) * (
                                                    b['Parallel_MMbond']==-1 ))
                            except KeyError: StillBoundAntiparallel.append(False)
                            try: StillBoundAntiparallelNotjammed.append(((dfPartM.loc[b.TopologyA]['Jammed']==False) * (dfPartM.loc[b.TopologyA]['Jammed']==False) * (dfPartM.loc[b.TopologyA]['MolecA_Mpart']==dfPartM_prts.loc[b.TopologyA]['MolecA_Mpart']) * (dfPartM.loc[b.TopologyA]['ParticleType']==8) * (dfPartM_prts.loc[b.TopologyA]['ParticleType']==8)) * (
                                                (dfPartM.loc[b.TopologyB]['MolecA_Mpart']==dfPartM_prts.loc[b.TopologyB]['MolecA_Mpart']) * (dfPartM.loc[b.TopologyB]['ParticleType']==8) * (dfPartM_prts.loc[b.TopologyB]['ParticleType']==8)) * (
                                                    b['Parallel_MMbond']==-1 ))
                            except KeyError: StillBoundAntiparallelNotjammed.append(False)
                            try: StillBoundParallel.append(((dfPartM.loc[b.TopologyA]['MolecA_Mpart']==dfPartM_prts.loc[b.TopologyA]['MolecA_Mpart']) * (dfPartM.loc[b.TopologyA]['ParticleType']==8) * (dfPartM_prts.loc[b.TopologyA]['ParticleType']==8)) * (
                                                (dfPartM.loc[b.TopologyB]['MolecA_Mpart']==dfPartM_prts.loc[b.TopologyB]['MolecA_Mpart']) * (dfPartM.loc[b.TopologyB]['ParticleType']==8) * (dfPartM_prts.loc[b.TopologyB]['ParticleType']==8)) * (
                                                    b['Parallel_MMbond']==1 ))
                            except KeyError: StillBoundParallel.append(False)
                            try: StillBoundParallelNotjammed.append(((dfPartM.loc[b.TopologyA]['Jammed']==False) * (dfPartM.loc[b.TopologyA]['Jammed']==False) * (dfPartM.loc[b.TopologyA]['MolecA_Mpart']==dfPartM_prts.loc[b.TopologyA]['MolecA_Mpart']) * (dfPartM.loc[b.TopologyA]['ParticleType']==8) * (dfPartM_prts.loc[b.TopologyA]['ParticleType']==8)) * (
                                                (dfPartM.loc[b.TopologyB]['MolecA_Mpart']==dfPartM_prts.loc[b.TopologyB]['MolecA_Mpart']) * (dfPartM.loc[b.TopologyB]['ParticleType']==8) * (dfPartM_prts.loc[b.TopologyB]['ParticleType']==8)) * (
                                                    b['Parallel_MMbond']==1 ))
                            except KeyError: StillBoundParallelNotjammed.append(False)
                            try: StillBoundParallelNotjammedTojammed.append(((dfPartM.loc[b.TopologyA]['Jammed']==False) * (dfPartM.loc[b.TopologyA]['Jammed']==False) * (dfPartM.loc[b.TopologyB]['Jammed']==True) * (dfPartM.loc[b.TopologyB]['Jammed']==True) * (dfPartM.loc[b.TopologyA]['MolecA_Mpart']==dfPartM_prts.loc[b.TopologyA]['MolecA_Mpart']) * (dfPartM.loc[b.TopologyA]['ParticleType']==8) * (dfPartM_prts.loc[b.TopologyA]['ParticleType']==8)) * (
                                                (dfPartM.loc[b.TopologyB]['MolecA_Mpart']==dfPartM_prts.loc[b.TopologyB]['MolecA_Mpart']) * (dfPartM.loc[b.TopologyB]['ParticleType']==8) * (dfPartM_prts.loc[b.TopologyB]['ParticleType']==8)) * (
                                                    b['Parallel_MMbond']==1 ))
                            except KeyError: StillBoundParallelNotjammedTojammed.append(False)
                            if timestep == timestep_prts:
                                MSpeed.append(0)
                            else: 
                                try: MSpeed.append((dfPartM.loc[b.TopologyA]['PolytimeA_Mpart']-dfPartM_prts.loc[b.TopologyA]['PolytimeA_Mpart'])/(timestep-timestep_prts))
                                except KeyError: MSpeed.append(np.nan)
                        else: 
                            MSpeed.append(0)
                            StillBoundAntiparallel.append(False)
                            StillBoundAntiparallelNotjammed.append(False)
                            StillBoundParallel.append(False)
                            StillBoundParallelNotjammed.append(False)
                            StillBoundParallelNotjammedTojammed.append(False)
                else:
                    assert ( dfPartM.loc[dfBondMM_ts.TopologyA]['ParticleIdentifier'].values == dfPartM_prts.loc[dfBondMM_prts.TopologyA]['ParticleIdentifier'].values ).all(), "Topology of MM bonds does not correspond to reference time frame"
                    dfBondMM_ts['Parallel_MMbond_prts'] = dfBondMM_prts['Parallel_MMbond']
                    # Double bound AND bound to same filaments AND antiparallel:
                    StillBoundAntiparallel = ((dfPartM.loc[dfBondMM_ts.TopologyA]['MolecA_Mpart']==dfPartM_prts.loc[dfBondMM_prts.TopologyA]['MolecA_Mpart']) * (dfPartM.loc[dfBondMM_ts.TopologyA]['ParticleType']==8) * (dfPartM_prts.loc[dfBondMM_prts.TopologyA]['ParticleType']==8)).values * (
                                                (dfPartM.loc[dfBondMM_ts.TopologyB]['MolecA_Mpart']==dfPartM_prts.loc[dfBondMM_prts.TopologyB]['MolecA_Mpart']) * (dfPartM.loc[dfBondMM_ts.TopologyB]['ParticleType']==8) * (dfPartM_prts.loc[dfBondMM_prts.TopologyB]['ParticleType']==8)).values * (
                                                    dfBondMM_ts['Parallel_MMbond']==-1 ).values
                    StillBoundAntiparallelNotjammed = (dfPartM.loc[dfBondMM_prts.TopologyA]['Jammed']==False).values*(dfPartM.loc[dfBondMM_ts.TopologyA]['Jammed']==False).values*(dfPartM.loc[dfBondMM_ts.TopologyA]['MolecA_Mpart']==dfPartM_prts.loc[dfBondMM_prts.TopologyA]['MolecA_Mpart']) * (dfPartM.loc[dfBondMM_ts.TopologyA]['ParticleType']==8) * (dfPartM_prts.loc[dfBondMM_prts.TopologyA]['ParticleType']==8).values * (
                                                (dfPartM.loc[dfBondMM_ts.TopologyB]['MolecA_Mpart']==dfPartM_prts.loc[dfBondMM_prts.TopologyB]['MolecA_Mpart']) * (dfPartM.loc[dfBondMM_ts.TopologyB]['ParticleType']==8) * (dfPartM_prts.loc[dfBondMM_prts.TopologyB]['ParticleType']==8)).values * (
                                                    dfBondMM_ts['Parallel_MMbond']==-1 ).values
                    # Double bound AND bound to same filaments AND parallel:
                    StillBoundParallel = ((dfPartM.loc[dfBondMM_ts.TopologyA]['MolecA_Mpart']==dfPartM_prts.loc[dfBondMM_prts.TopologyA]['MolecA_Mpart']) * (dfPartM.loc[dfBondMM_ts.TopologyA]['ParticleType']==8) * (dfPartM_prts.loc[dfBondMM_prts.TopologyA]['ParticleType']==8)).values * (
                                                (dfPartM.loc[dfBondMM_ts.TopologyB]['MolecA_Mpart']==dfPartM_prts.loc[dfBondMM_prts.TopologyB]['MolecA_Mpart']) * (dfPartM.loc[dfBondMM_ts.TopologyB]['ParticleType']==8) * (dfPartM_prts.loc[dfBondMM_prts.TopologyB]['ParticleType']==8)).values * (
                                                    dfBondMM_ts['Parallel_MMbond']==1 ).values
                    StillBoundParallelNotjammed = (dfPartM.loc[dfBondMM_prts.TopologyA]['Jammed']==False).values*(dfPartM.loc[dfBondMM_ts.TopologyA]['Jammed']==False).values*(dfPartM.loc[dfBondMM_ts.TopologyA]['MolecA_Mpart']==dfPartM_prts.loc[dfBondMM_prts.TopologyA]['MolecA_Mpart']) * (dfPartM.loc[dfBondMM_ts.TopologyA]['ParticleType']==8) * (dfPartM_prts.loc[dfBondMM_prts.TopologyA]['ParticleType']==8).values * (
                                                (dfPartM.loc[dfBondMM_ts.TopologyB]['MolecA_Mpart']==dfPartM_prts.loc[dfBondMM_prts.TopologyB]['MolecA_Mpart']) * (dfPartM.loc[dfBondMM_ts.TopologyB]['ParticleType']==8) * (dfPartM_prts.loc[dfBondMM_prts.TopologyB]['ParticleType']==8)).values * (
                                                    dfBondMM_ts['Parallel_MMbond']==1 ).values
                    StillBoundParallelNotjammedTojammed = (dfPartM.loc[dfBondMM_prts.TopologyA]['Jammed']==False).values*(dfPartM.loc[dfBondMM_prts.TopologyB]['Jammed']==True).values*(dfPartM.loc[dfBondMM_ts.TopologyA]['Jammed']==False).values*(dfPartM.loc[dfBondMM_ts.TopologyB]['Jammed']==True).values*(dfPartM.loc[dfBondMM_ts.TopologyA]['MolecA_Mpart']==dfPartM_prts.loc[dfBondMM_prts.TopologyA]['MolecA_Mpart']) * (dfPartM.loc[dfBondMM_ts.TopologyA]['ParticleType']==8) * (dfPartM_prts.loc[dfBondMM_prts.TopologyA]['ParticleType']==8).values * (
                                                (dfPartM.loc[dfBondMM_ts.TopologyB]['MolecA_Mpart']==dfPartM_prts.loc[dfBondMM_prts.TopologyB]['MolecA_Mpart']) * (dfPartM.loc[dfBondMM_ts.TopologyB]['ParticleType']==8) * (dfPartM_prts.loc[dfBondMM_prts.TopologyB]['ParticleType']==8)).values * (
                                                    dfBondMM_ts['Parallel_MMbond']==1 ).values
                    # Compute speed wrt reference
                    if timestep==timestep_prts:
                        MSpeed = np.zeros(len(dfBondMM_ts))
                    else:
                        MSpeed = (dfPartM.loc[dfBondMM_ts.TopologyA]['PolytimeA_Mpart']-dfPartM_prts.loc[dfBondMM_prts.TopologyA]['PolytimeA_Mpart'])/(timestep-timestep_prts)

                dfBondMM_ts['StillBoundAntiparallel'] = StillBoundAntiparallel
                dfBondMM_ts['StillBoundAntiparallelNotjammed'] = StillBoundAntiparallelNotjammed
                dfBondMM_ts['StillBoundParallel'] = StillBoundParallel
                dfBondMM_ts['StillBoundParallelNotjammed'] = StillBoundParallelNotjammed
                dfBondMM_ts['StillBoundParallelNotjammedTojammed'] = StillBoundParallelNotjammedTojammed #!
                dfBondMM_ts['MSpeed'] = MSpeed

                dfBondMM_ts_Antiparallel = dfBondMM_ts[dfBondMM_ts['StillBoundAntiparallel']==True]
                dfBondMM_ts_Parallel = dfBondMM_ts[dfBondMM_ts['StillBoundParallel']==True]
                dfBondMM_ts_Antiparallel_Notjammed = dfBondMM_ts[dfBondMM_ts['StillBoundAntiparallelNotjammed']==True]
                dfBondMM_ts_Parallel_Notjammed = dfBondMM_ts[dfBondMM_ts['StillBoundParallelNotjammed']==True]
                dfBondMM_ts_Parallel_Notjammed_Tojammed = dfBondMM_ts[dfBondMM_ts['StillBoundParallelNotjammedTojammed']==True] #!
                dfPartM_prts=dfPartM
                dfBondMM_prts=dfBondMM_ts
                timestep_prts = timestep
                # store data
                MSpeedAntiparallelAvg = np.nanmean(dfBondMM_ts_Antiparallel['MSpeed'])
                MSpeedParallelAvg  = np.nanmean(dfBondMM_ts_Parallel['MSpeed'])
                MSpeedAntiparallelNotjammedAvg = np.nanmean(dfBondMM_ts_Antiparallel_Notjammed['MSpeed'])
                MSpeedParallelNotjammedAvg = np.nanmean(dfBondMM_ts_Parallel_Notjammed['MSpeed'])
                if len(dfBondMM_ts_Parallel_Notjammed_Tojammed)>0: MSpeedParallelNotjammedTojammedAvg = np.nanmean(dfBondMM_ts_Parallel_Notjammed_Tojammed['MSpeed'])
                else: MSpeedParallelNotjammedTojammedAvg = np.nan
                if len(dfBondMM_ts_Parallel_Notjammed_Tojammed) == 0: MSpeedRelevantAvg = MSpeedAntiparallelNotjammedAvg
                elif len(dfBondMM_ts_Antiparallel_Notjammed) == 0: MSpeedRelevantAvg = MSpeedAntiparallelNotjammedAvg
                else: MSpeedRelevantAvg = (MSpeedAntiparallelNotjammedAvg*len(dfBondMM_ts_Antiparallel_Notjammed)+MSpeedParallelNotjammedTojammedAvg*len(dfBondMM_ts_Parallel_Notjammed_Tojammed))/(len(dfBondMM_ts_Antiparallel_Notjammed)+len(dfBondMM_ts_Parallel_Notjammed_Tojammed))
                MSpeedAvg = (MSpeedAntiparallelAvg*len(dfBondMM_ts_Parallel)+MSpeedParallelAvg*len(dfBondMM_ts_Antiparallel))/(len(dfBondMM_ts_Parallel)+len(dfBondMM_ts_Antiparallel)) if (len(dfBondMM_ts_Parallel)+len(dfBondMM_ts_Antiparallel)) != 0 else np.nan
                MSpeedNotjammedAvg = (MSpeedAntiparallelNotjammedAvg*len(dfBondMM_ts_Parallel_Notjammed)+MSpeedParallelNotjammedAvg*len(dfBondMM_ts_Antiparallel_Notjammed))/(len(dfBondMM_ts_Parallel_Notjammed)+len(dfBondMM_ts_Antiparallel_Notjammed)) if (len(dfBondMM_ts_Parallel_Notjammed)+len(dfBondMM_ts_Antiparallel_Notjammed)) != 0 else np.nan
                histrange = [-(lF+0.5)/AnalyseEvery, (lF+0.5)/AnalyseEvery]  if timestep==timestep_prts  else [-(lF+0.5)/(timestep-timestep_prts), (lF+0.5)/(timestep-timestep_prts)]
                hist, bins = np.histogram(dfBondMM_ts_Antiparallel['MSpeed'], bins=2*lF+1, range=histrange, density=False)
                histMSpeedAntiparallel = [bins,hist]
                hist, bins = np.histogram(dfBondMM_ts_Parallel['MSpeed'], bins=2*lF+1, range=histrange, density=False)
                histMSpeedParallel = [bins,hist]
                hist, bins = np.histogram(dfBondMM_ts_Antiparallel_Notjammed['MSpeed'], bins=2*lF+1, range=histrange, density=False)
                histMSpeedAntiparallelNotjammed = [bins,hist]
                hist, bins = np.histogram(dfBondMM_ts_Parallel_Notjammed['MSpeed'], bins=2*lF+1, range=histrange, density=False)
                histMSpeedParallelNotjammed = [bins,hist]
                line.append(MSpeedAntiparallelAvg)
                linecolumns.append('MSpeedAntiparallelAvg')
                line.append(MSpeedParallelAvg)
                linecolumns.append('MSpeedParallelAvg')
                line.append(MSpeedAntiparallelNotjammedAvg)
                linecolumns.append('MSpeedAntiparallelNotjammedAvg')
                line.append(MSpeedParallelNotjammedAvg)
                linecolumns.append('MSpeedParallelNotjammedAvg')
                line.append(histMSpeedAntiparallel)
                linecolumns.append('histMSpeedAntiparallel')
                line.append(histMSpeedParallel)
                linecolumns.append('histMSpeedParallel')
                line.append(histMSpeedAntiparallelNotjammed)
                linecolumns.append('histMSpeedAntiparallelNotjammed')
                line.append(histMSpeedParallelNotjammed)
                linecolumns.append('histMSpeedParallelNotjammed')
                line.append(MSpeedAvg)
                linecolumns.append('MSpeedAvg')
                line.append(MSpeedNotjammedAvg)
                linecolumns.append('MSpeedNotjammedAvg')
                line.append(MSpeedRelevantAvg)
                linecolumns.append('MSpeedRelevantAvg')
                print(' MSpeedParallelAvg is {:g}, MSpeedAntiparallelAvg is {:g}'.format(MSpeedParallelAvg,MSpeedAntiparallelAvg))
                print(' MSpeedParallelNotjammedAvg is {:g}, MSpeedAntiparallelNotjammedAvg is {:g}'.format(MSpeedParallelNotjammedAvg,MSpeedAntiparallelNotjammedAvg))
                print("Parallel not jammed to jammed: "+str(len(dfBondMM_ts_Parallel_Notjammed_Tojammed)))
                print("Antiparallel not jammed: "+str(len(dfBondMM_ts_Antiparallel_Notjammed)))
                print(' MSpeedRelevantAvg is {:g}'.format(MSpeedRelevantAvg))
            
            # histogram of contacts between parallel and antiparallel molecules
            if np.array([x in PropertiesToCompute for x in ['histParallelContacts','histAntiparallelExtensileContacts', 'histAntiparallelContractileContacts']]).any() and not sector:
                            # adfOldtimestep not in dfOld['time']):
                print(" Contacts")
                ThetaScreeningTolerance = np.pi/18 # particles considered possible neighbours if closer than 10 deg
                Distance2Tolerance = (2)**2 # It used to be MMbondLength+3, I changed it to 2, so that it counts really contacts
                MolecList = dfPartA.Molec.unique()
                ContactData = [[],[],[]]
                for jm1, m1 in enumerate(MolecList):
                    dfm1 = dfPartA[dfPartA['Molec']==m1]
                    #dfm1 = addFilamentDirection(dfm1,Lz)
                    Dir1 = dfm1.DirectionAA_Apart.mean() #globalFilamentDirection(dfm1, Lz)
                    Positions = np.array([[x[0],x[1],x[2]] for x in dfm1.Position.values])
                    dm1 = np.array([dfm1.ParticleIdentifier.values, Positions[:,0], Positions[:,1], Positions[:,2], dfm1.ThetaRad.values]).transpose()
                    for m2 in MolecList[(jm1+1):]:
                        dfm2 = dfPartA[dfPartA['Molec']==m2]
                        if ThetaOverlap(dfm1.ThetaRad.values, dfm2.ThetaRad.values, 0.1):
                            # dfm2 = addFilamentDirection(dfm2,Lz)
                            Dir2 = dfm2.DirectionAA_Apart.mean() #globalFilamentDirection(dfm2, Lz)
                            Positions = np.array([[x[0],x[1],x[2]] for x in dfm2.Position.values])
                            dm2 = np.array([dfm2.ParticleIdentifier.values, Positions[:,0], Positions[:,1], Positions[:,2], dfm2.ThetaRad.values]).transpose()
                            ContactCounter = countcontacts(dm1,dm2,Distance2Tolerance,ThetaScreeningTolerance,Dir1,Dir2,lF)
                            for ContactFlag in [0,1,2]:
                                if ContactCounter[ContactFlag]>0:
                                    ContactData[ContactFlag].append(ContactCounter[ContactFlag])

                # parallel
                hist, bins = np.histogram(ContactData[0], bins=lF, range=[0,lF], density=False)
                histParallelContactsThisFrame = [bins, hist]
                # antiparallel extensile
                hist, bins = np.histogram(ContactData[1], bins=lF, range=[0,lF], density=False)
                histAntiparallelExtensileContactsThisFrame = [bins, hist]
                # antiparallel contractile
                hist, bins = np.histogram(ContactData[2], bins=lF, range=[0,lF], density=False)
                histAntiparallelContractileContactsThisFrame = [bins, hist]
            elif 'histParallelContacts' in dfOld_ts.keys() :
                histParallelContactsThisFrame, histAntiparallelExtensileContactsThisFrame, histAntiparallelContractileContactsThisFrame = dfOld_ts['histParallelContacts'], dfOld_ts['histAntiparallelExtensileContacts'], dfOld_ts['histAntiparallelContractileContacts']
            else:
                histParallelContactsThisFrame, histAntiparallelExtensileContactsThisFrame, histAntiparallelContractileContactsThisFrame = [],[],[]
            if len(histParallelContactsThisFrame)>0:
                line.append(histParallelContactsThisFrame)
                linecolumns.append('histParallelContacts')
                line.append(histAntiparallelExtensileContactsThisFrame)
                linecolumns.append('histAntiparallelExtensileContacts')
                line.append(histAntiparallelContractileContactsThisFrame)
                linecolumns.append('histAntiparallelContractileContacts')

            # Compute myosin contact map
            if np.array([x in PropertiesToCompute for x in ['histParallelMbond','histAntiparallelMbond']]).any() and not sector:
                hist, bins = np.histogram(dfBondMM[dfBondMM['Parallel_MMbond']==-1]['Extensivity_MMbond'], bins=2*lF+1, range=[-lF-0.5,lF+0.5], density=False)
                histAntiparallelMbondThisFrame = [bins,hist]
                hist, bins = np.histogram(dfBondMM[dfBondMM['Parallel_MMbond']==-1]['StaggerStuck_MMbond'], bins=lF+2, range=[-1.5,lF+0.5], density=False)
                histAntiparallelMbondThisFrame.append(bins)
                histAntiparallelMbondThisFrame.append(hist)
                hist, bins = np.histogram(dfBondMM[dfBondMM['Parallel_MMbond']==1]['Extensivity_MMbond'], bins=2*lF+1, range=[-lF-0.5-1000,lF+0.5-1000], density=False)
                histParallelMbondThisFrame = [ [x+1000 for x in bins], hist]
                hist, bins = np.histogram(dfBondMM[dfBondMM['Parallel_MMbond']==1]['StaggerStuck_MMbond'], bins=lF+2, range=[-1.5,lF+0.5], density=False)
                histParallelMbondThisFrame.append(bins)
                histParallelMbondThisFrame.append(hist)
                # histParallelMbondThisFrame columns=['binsExtensivity', 'histExtensivity', 'binsStaggerStack', 'histStaggerStuck'])
            elif np.array([x in dfOld_ts.keys() for x in ['histParallelMbond','histAntiparallelMbond']]).all() and not sector:
                histParallelMbondThisFrame, histAntiparallelMbondThisFrame = dfOld_ts['histParallelMbond'], dfOld_ts['histAntiparallelMbond']
            else:
                histParallelMbondThisFrame, histAntiparallelMbondThisFrame = [], []
            if len(histParallelMbondThisFrame)>0:
                line.append(histParallelMbondThisFrame)
                linecolumns.append('histParallelMbond')
                line.append(histAntiparallelMbondThisFrame)
                linecolumns.append('histAntiparallelMbond')

            # Compute advancement in extensivity in time, wrt to previous timestep, considering only myosins that stay bounded to the 2 filaments to which they are bound at the reference time step.
            # I.e. how much do myosins contribute to contraction/extension before coming off?
            if np.array([x in PropertiesToCompute for x in ['DeltaExtensivityRef']]).any() and not sector:
                print(" DeltaExtensivityRef")
                #By how much myosins have moved, wrt to timestep say 1000000, as a function of Extensivity_MMbond, parallel/antiparallel state and one end being bound to barbed end or not
                ReferenceTimestep = 1000000
                dfBondMM_ts = dfBondMM.sort_values(by='TopologyA').set_index('TopologyA',drop=False)
                for c in ['Length','Energy','PolytimeA_AMbond','MolecA_AMbond']:
                    del dfBondMM_ts[c]
                if timestep==ReferenceTimestep:
                    dfPartM_Reference=dfPartM
                    dfBondMM_Reference=dfBondMM_ts
                    dfPartM_prets=dfPartM
                    dfBondMM_prets=dfBondMM_ts
                if timestep>=ReferenceTimestep and timestep<=ReferenceTimestep+2000000:
                    assert ( dfPartM.loc[dfBondMM_ts.TopologyA]['ParticleIdentifier'].values == dfPartM_Reference.loc[dfBondMM_Reference.TopologyA]['ParticleIdentifier'].values ).all(), "Topology of MM bonds does not correspond to reference time frame"
                    dfBondMM_ts['Extensivity_MMbond_Reference'] = dfBondMM_Reference['Extensivity_MMbond']
                    # Double bound AND still bound to same filaments AND antiparallel:
                    StillBound = ((dfPartM.loc[dfBondMM_ts.TopologyA]['MolecA_Mpart']==dfPartM_Reference.loc[dfBondMM_Reference.TopologyA]['MolecA_Mpart']) * (dfPartM.loc[dfBondMM_ts.TopologyA]['ParticleType']==8)).values * (
                        (dfPartM.loc[dfBondMM_ts.TopologyB]['MolecA_Mpart']==dfPartM_Reference.loc[dfBondMM_Reference.TopologyB]['MolecA_Mpart']) * (dfPartM.loc[dfBondMM_ts.TopologyB]['ParticleType']==8)).values * (
                        dfBondMM_ts['Extensivity_MMbond']>=-(lF+1) ).values
                    if timestep==ReferenceTimestep:
                        dfBondMM_Reference['StillBound'] = StillBound
                    else:
                        dfBondMM_Reference['StillBound'] = dfBondMM_Reference['StillBound'] * StillBound
                    # Compute DeltaExtensivity wrt reference
                    dfBondMM_ts['DeltaExtensivity_MMbond'] = (dfBondMM_ts['Extensivity_MMbond'] - dfBondMM_prets['Extensivity_MMbond']) * dfBondMM_Reference['StillBound'].astype(int) - 1000 *(1-dfBondMM_Reference['StillBound'].astype(int))
                    dfBondMM_ts_stripped = dfBondMM_ts[dfBondMM_Reference['StillBound']==True]
                    dfmean = dfBondMM_ts_stripped.groupby(['Extensivity_MMbond_Reference']).mean(numeric_only=True)
                    dfstd = dfBondMM_ts_stripped.groupby(['Extensivity_MMbond_Reference']).std(numeric_only=True)
                    dfsum = dfBondMM_ts_stripped.groupby(['Extensivity_MMbond_Reference']).sum(numeric_only=True)
                    dfcount = dfBondMM_ts_stripped.groupby(['Extensivity_MMbond_Reference']).count()
                    InitialExtensivity=np.arange(-lF,lF+1,1)
                    DeltaMean = []
                    DeltaStd = []
                    DeltaSum = []
                    DeltaCount = []
                    for x in InitialExtensivity:
                        try:
                            DeltaMean.append(dfmean.loc[x].DeltaExtensivity_MMbond)
                            DeltaStd.append(dfstd.loc[x].DeltaExtensivity_MMbond)
                            DeltaSum.append(dfsum.loc[x].DeltaExtensivity_MMbond)
                            DeltaCount.append(dfcount.loc[x].DeltaExtensivity_MMbond)
                        except:
                            DeltaMean.append(-1)
                            DeltaStd.append(-1)
                            DeltaSum.append(0)
                            DeltaCount.append(0)
                    DeltaExtensivityRefThisFrame = pd.DataFrame(np.array([InitialExtensivity,DeltaMean,DeltaStd,DeltaSum,DeltaCount]).transpose(), columns=['InitialExtensivity','DeltaMean','DeltaStd','DeltaSum','DeltaCount'])
                    dfPartM_prets=dfPartM
                    dfBondMM_prets=dfBondMM_ts
                else:
                    DeltaExtensivityRefThisFrame = []
            elif 'DeltaExtensivityRef' in dfOld_ts.keys():
                DeltaExtensivityRefThisFrame = dfOld_ts['DeltaExtensivityRef']
            else:
                DeltaExtensivityRefThisFrame = []
            if len(DeltaExtensivityRefThisFrame)>0:
                line.append(DeltaExtensivityRefThisFrame)
                linecolumns.append('DeltaExtensivityRef')

            # Compute DeltaExtensivity in time, wrt to previous frame, considering all myosins
            if np.array([x in PropertiesToCompute for x in ['DeltaExtensivityCont']]).any() and not sector:

                #By how much myosins have moved, wrt to timestep say 1000000, as a function of Extensivity_MMbond, parallel/antiparallel state and one end being bound to barbed end or not
                dfBondMM_ts = dfBondMM.sort_values(by='TopologyA').set_index('TopologyA',drop=False)
                for c in ['Length','Energy','PolytimeA_AMbond','MolecA_AMbond']:
                    del dfBondMM_ts[c]
                if len(dfPartM_prevts)==0:
                    dfPartM_prevts=dfPartM
                    dfBondMM_prevts=dfBondMM_ts
                assert ( dfPartM.loc[dfBondMM_ts.TopologyA]['ParticleIdentifier'].values == dfPartM_prevts.loc[dfBondMM_prevts.TopologyA]['ParticleIdentifier'].values ).all(), "Topology of MM bonds does not correspond to reference time frame"
                dfBondMM_ts['Extensivity_MMbond_prevts'] = dfBondMM_prevts['Extensivity_MMbond']
                # Double bound AND bound to same filaments AND antiparallel:
                StillBound = ((dfPartM.loc[dfBondMM_ts.TopologyA]['MolecA_Mpart']==dfPartM_prevts.loc[dfBondMM_prevts.TopologyA]['MolecA_Mpart']) * (dfPartM.loc[dfBondMM_ts.TopologyA]['ParticleType']==8) * (dfPartM_prevts.loc[dfBondMM_prevts.TopologyA]['ParticleType']==8)).values * (
                    (dfPartM.loc[dfBondMM_ts.TopologyB]['MolecA_Mpart']==dfPartM_prevts.loc[dfBondMM_prevts.TopologyB]['MolecA_Mpart']) * (dfPartM.loc[dfBondMM_ts.TopologyB]['ParticleType']==8) * (dfPartM_prevts.loc[dfBondMM_prevts.TopologyB]['ParticleType']==8)).values * (
                        dfBondMM_ts['Extensivity_MMbond']>=-(lF+1) ).values
                dfBondMM_ts['StillBound'] = StillBound
                # Compute DeltaExtensivity wrt reference
                dfBondMM_ts['DeltaExtensivity_MMbond'] = (dfBondMM_ts['Extensivity_MMbond'] - dfBondMM_prevts['Extensivity_MMbond']) * dfBondMM_ts['StillBound'].astype(int) - 1000 *(1-dfBondMM_ts['StillBound'].astype(int))
                dfBondMM_ts_stripped = dfBondMM_ts[dfBondMM_ts['StillBound']==True]
                dfmean = dfBondMM_ts_stripped.groupby(['Extensivity_MMbond_prevts']).mean(numeric_only=True)
                dfstd = dfBondMM_ts_stripped.groupby(['Extensivity_MMbond_prevts']).std(numeric_only=True)
                dfsum = dfBondMM_ts_stripped.groupby(['Extensivity_MMbond_prevts']).sum(numeric_only=True)
                dfcount = dfBondMM_ts_stripped.groupby(['Extensivity_MMbond_prevts']).count()
                PreviousExtensivity=np.arange(-lF,lF+1,1)
                DeltaMean = []
                DeltaStd = []
                DeltaSum = []
                DeltaCount = []
                for x in PreviousExtensivity:
                    try:
                        DeltaMean.append(dfmean.loc[x].DeltaExtensivity_MMbond)
                        DeltaStd.append(dfstd.loc[x].DeltaExtensivity_MMbond)
                        DeltaSum.append(dfsum.loc[x].DeltaExtensivity_MMbond)
                        DeltaCount.append(dfcount.loc[x].DeltaExtensivity_MMbond)
                    except:
                        DeltaMean.append(-1)
                        DeltaStd.append(-1)
                        DeltaSum.append(0)
                        DeltaCount.append(0)
                DeltaExtensivityContThisFrame = pd.DataFrame(np.array([PreviousExtensivity,DeltaMean,DeltaStd,DeltaSum,DeltaCount]).transpose(), columns=['PreviousExtensivity','DeltaMean','DeltaStd','DeltaSum','DeltaCount'])
                dfPartM_prevts=dfPartM
                dfBondMM_prevts=dfBondMM_ts
            elif 'DeltaExtensivityCont' in dfOld_ts.keys():
                DeltaExtensivityContThisFrame = dfOld_ts['DeltaExtensivityCont']
            else:
                DeltaExtensivityContThisFrame = []
            if len(DeltaExtensivityContThisFrame)>0:
                line.append(DeltaExtensivityContThisFrame)
                linecolumns.append('DeltaExtensivityCont')

            ############################################
            # put everyting in one line of the dataframe
            linesPerFrame.append(line)
            prevtimestep=timestep
            #print(' {:d}-{:d}'.format(len(line),len(linecolumns)))
    return linesPerFrame, linecolumns

Properties = ['time', 'Ravg', 'Rstd', 'histAdensity', 'histABdensity', 'histAPdensity', 'histMUdensity', 'histMBdensity', 
              'RingIntactFlag', 'PercolatedFraction',#'histParallelMbond', 'histAntiparallelMbond',
              'ActinStrainAvg','histActinStrainDistr','histMDistrAlongActin','histDoubleBoundMDistrAlongActin',#'vdotnActinAvg', 'histvdotnActinDistr',
              'MSpeedParallelAvg','MSpeedAntiparallelAvg','MSpeedAvg','MSpeedNotjammedAvg',
              'JammedMyosins','JammedActins','NotjammedDoubleBoundMyosins','EndtoendDistance','MyosinStrainAvg'
             ]

for filestring in filestringList[:]:
    print(filestring)
    if filestring in oldfiles:
        print('oldfile')
        AnalyseEvery = 300000
        TimestepInterval = 30000
    elif filestring in middlefiles:
        print('middlefile')
        AnalyseEvery = 300000
        TimestepInterval = 50000
    elif filestring in finefiles:
        print('finefile')
        AnalyseEvery = 300000
        TimestepInterval = 10000
    elif filestring in longfiles:
        print('longfile')
        AnalyseEvery = 600000
        TimestepInterval = 200000
    else:
        AnalyseEvery = 300000
        TimestepInterval = 100000 #100000

    lastoutstring = subprocess.check_output("tail -n5 {:s}/Results/Log_{:s}.dat".format(folder,filestring), shell=True)
    #if (os.path.isfile(ClusterStatisticsFile)==False) & ("Total wall time" in laststring.decode("utf-8")):
    if True: #("Total wall time" in lastoutstring.decode("utf-8")):
        # Data import:
        pipeline, NumberOfFrames, PropertiesToCompute  = setup_pipeline(filestring,Properties, TimestepInterval, AnalyseEvery)

        # Viewport:
        #vp = Viewport(type = Viewport.Type.Ortho,fov = 125.553126564,camera_dir = (-0.217381977313889, 0.5270208842467977, -0.8215802234151132),camera_pos = (-1.0869218425781972, 2.6351334073826225, 0.8921086960121469))

        # Analyse frames:
        if NumberOfFrames!= 0:
            try: linesPerFrame, linecolumns = analyse_frames(getparams(filestring),pipeline, NumberOfFrames, PropertiesToCompute,TimestepInterval, AnalyseEvery)
            except RuntimeError: 
                print("There was some RunTimeError!\n")
                continue
        else: continue
        # Fill dataframe
        if len(linesPerFrame)>0:
            df = pd.DataFrame(linesPerFrame, columns=linecolumns)
            dicR[filestring] = df
        
        # Update/create this files's pickle entry
        with open('{:s}/Analysis/dicR.pickle'.format(folder), 'wb') as file:
            pickle.dump(dicR, file)

        """print("now the sector")
        pipeline, NumberOfFrames, PropertiesToCompute  = setup_pipeline(filestring,Properties, 1000, 10000, sector=True)

        # Analyse frames:
        if NumberOfFrames!= 0:
            try: linesPerFrame, linecolumns = analyse_frames(getparams(filestring),pipeline, NumberOfFrames, PropertiesToCompute, 1000,10000, sector = True)
            except RuntimeError: 
                print("There was some RunTimeError!\n")
                continue
        else: continue

        # Fill dataframe
        if len(linesPerFrame)>0:
            df = pd.DataFrame(linesPerFrame, columns=linecolumns)
            dicR['Sector_'+filestring] = df
        
        # Update/create this files's pickle entry
        with open('{:s}/Analysis/dicR.pickle'.format(folder), 'wb') as file:
            pickle.dump(dicR, file)"""

        
