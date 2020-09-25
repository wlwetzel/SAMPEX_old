import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from scipy import signal
sys.path.append('/home/wyatt/pyModules')
import SAMP_Data
import SAMPEXreb
import datetime
from spacepy.time import Ticktock
import spacepy.coordinates as spc
import spacepy.irbempy as irb
import matplotlib.animation as animation
from Wavelet import wavelet,wave_signif,invertWave
from scipy.integrate import simps
from pitchAngle_v4 import *
import os
Re = 6371 #km
"""
This script is for analysis of identified microbursts
1) read in microbursts
2) find local pitch angle range
"""
filename = '/home/wyatt/Documents/SAMPEX/generated_Data/procData2.csv'
# if os.path.exists(filename):
#   os.remove(filename)
# else:
#   print("The file does not exist")

radToDeg = 180.0 / m.pi
cols = ['Rate1', 'Rate2', 'Rate3', 'Rate4', 'GEI_X', 'GEI_Y', 'GEI_Z', 'A11', 'A21', 'A31', 'A12', 'A22', 'A32', 'A13', 'A23', 'A33']

data = pd.read_csv('/home/wyatt/Documents/SAMPEX/generated_Data/burst_Jan93.csv',header=None,index_col=0,names=cols)
data2 = pd.read_csv('/home/wyatt/Documents/SAMPEX/generated_Data/burst_Feb93.csv',header=None,index_col=0,names=cols)
data3 = pd.read_csv('/home/wyatt/Documents/SAMPEX/generated_Data/burst_Mar93.csv',header=None,index_col=0,names=cols)
data = data.append(data2)
data = data.append(data3)
data = data.loc[~data.index.duplicated(keep='first')]
length = len(data.index.values)
times = data.index
print(length)
X = data['GEI_X'].to_numpy() / Re
Y = data['GEI_Y'].to_numpy() / Re
Z = data['GEI_Z'].to_numpy() / Re
position = np.stack((X,Y,Z),axis=1)
ticks = Ticktock(times)
coords = spc.Coords(position,'GEI','car')

#find B based on position
bField = irb.get_Bfield(ticks,coords,extMag='T89')
Field_GEO = bField['Bvec'] #b field comes out in xyz
#convert to GEIcar, because the direction cosines take GEI to body fixed
Field_GEO_coords = spc.Coords(Field_GEO,'GEO','car')
Field_GEO_coords.ticks = Ticktock(times)
Field_GEI = irb.coord_trans(Field_GEO_coords,'GEI','car')
Field_Mag = bField['Blocal']
#rotate b to body fixed
BX = Field_GEI[:,0]
BY = Field_GEI[:,1]
BZ = Field_GEI[:,2]

A11 = data['A11']
A12 = data['A12']
A13 = data['A13']
A21 = data['A21']
A22 = data['A22']
A23 = data['A23']
A31 = data['A31']
A32 = data['A32']
A33 = data['A33']

#x comp
BX_body = BX * A11 + BY * A12 + BZ * A13
#y comp
BY_body = BX * A21 + BY * A22 + BZ * A23
#z comp
BZ_body = BX * A31 + BY * A32 + BZ * A33
# octant = pd.Series(data=BZ_body)

#angle to yz plane
#alpha and beta
# if we take away the abs, we'll get the right sign
yz_angle = radToDeg * np.arcsin(BX_body / Field_Mag)

#angle to xz plane
#assoc. with -34 to 34
xz_angle = radToDeg * np.arcsin(BY_body / Field_Mag)

# we need to set up the ranges
leftEdge_xz = (-34 - xz_angle)
rightEdge_xz =( 34 - xz_angle)

#note, I think I calculated this wrong previously whoops
#det1 goes from -9.57 to 34
#det2 -18.636 o 26.834
#det3 -26.834 to 18.636
#det4 -34 to 9.57
leftEdge_1_yz = (-9.57 - yz_angle)
rightEdge_1_yz =(34 - yz_angle)

leftEdge_2_yz = (-18.636 - yz_angle)
rightEdge_2_yz = (26.834 - yz_angle)

leftEdge_3_yz = (-26.834 - yz_angle)
rightEdge_3_yz =( 18.636 - yz_angle)

leftEdge_4_yz = (-34 - yz_angle)
rightEdge_4_yz = (9.57 - yz_angle)

def fov_to_pitch(mesh):
    """
    for taking the square fov we have with respect to sampex coordinate planes
    and converting to local pitch angle
    """
    return np.rad2deg(np.arctan(np.sqrt(np.tan( np.deg2rad(mesh[0]) )**2 + np.tan( np.deg2rad(mesh[1]) )**2)))

def to_equatorial(position,time,pitch):
    """
    take in spacepy coord class and ticktock class
    """

    blocal = irb.get_Bfield(time,position,extMag='T89')['Blocal']
    beq = irb.find_magequator(time,position,extMag='T89')['Bmin']
    eq_pitch = np.arcsin(np.sqrt(np.sin(np.deg2rad(pitch))**2 * beq / blocal))
    return np.rad2deg(eq_pitch)

def find_loss_cone(position,time):
    foot = irb.find_footpoint(time,position,extMag='T89')['Bfoot']
    eq = irb.find_magequator(time,position,extMag='T89')['Bmin']
    pitch=90 #for particles mirroring at 100km
    return np.rad2deg(np.arcsin(np.sqrt(np.sin(np.deg2rad(pitch))**2 * eq / foot)))

# i think the next bit, converting to pitch angle, will be easiest in a for loop
for i in range(length):
    print(i)
    num = 10
    mesh1 = np.meshgrid(np.linspace(leftEdge_1_yz[i] , rightEdge_1_yz[i],num) ,
                        np.linspace(leftEdge_xz[i] , rightEdge_xz[i] , num))
    mesh2 = np.meshgrid(np.linspace(leftEdge_2_yz[i] , rightEdge_2_yz[i],num) ,
                        np.linspace(leftEdge_xz[i] , rightEdge_xz[i] , num))
    mesh3 = np.meshgrid(np.linspace(leftEdge_3_yz[i] , rightEdge_3_yz[i],num) ,
                        np.linspace(leftEdge_xz[i] , rightEdge_xz[i] , num))
    mesh4 = np.meshgrid(np.linspace(leftEdge_4_yz[i] , rightEdge_4_yz[i],num) ,
                        np.linspace(leftEdge_xz[i], rightEdge_xz[i] , num))

    #convert to pitchangle
    pitch1 = fov_to_pitch(mesh1).flatten()
    pitch2 = fov_to_pitch(mesh2).flatten()
    pitch3 = fov_to_pitch(mesh3).flatten()
    pitch4 = fov_to_pitch(mesh4).flatten()

    #going to try a different way of divvying up pitch angles
    # pitch1 = np.linspace(np.min(pitch1),np.max(pitch1) , num)
    # pitch2 = np.linspace(np.min(pitch2),np.max(pitch2) , num)
    # pitch3 = np.linspace(np.min(pitch3),np.max(pitch3) , num)
    # pitch4 = np.linspace(np.min(pitch4),np.max(pitch4) , num)

    #loss cone
    loss = find_loss_cone(coords[i],Ticktock(times[i]))

    pitch1 = to_equatorial(coords[i],Ticktock(times[i]),pitch1)
    pitch2 = to_equatorial(coords[i],Ticktock(times[i]),pitch2)
    pitch3 = to_equatorial(coords[i],Ticktock(times[i]),pitch3)
    pitch4 = to_equatorial(coords[i],Ticktock(times[i]),pitch4)
    np.set_printoptions(precision=3)
    # now we can just assign each element of the pitches a hundredth of the flux
    flux1 = np.array([data['Rate1'].iloc[i] / num**2 ] * num**2)
    flux2 = np.array([data['Rate2'].iloc[i] / num**2 ] * num**2)
    flux3 = np.array([data['Rate3'].iloc[i] / num**2 ] * num**2)
    flux4 = np.array([data['Rate4'].iloc[i] / num**2 ] * num**2)
    print(loss)
    pitches = np.append(pitch1, (pitch2,pitch3,pitch4))
    fluxes = np.append(flux1, (flux2,flux3,flux4))
    loss = np.array([loss]*len(pitches))
    loss=loss.flatten()
    df = pd.DataFrame(data={'pitch':pitches, 'flux':fluxes,'loss':loss})
    with open(filename,'a') as f:
        df.to_csv(f,header=None)



#what quadrant are we in
# for i in range(length):
#     if BX_body.iloc[i]>0:
#         if BY_body.iloc[i]>0:
#             if BZ_body.iloc[i]>0:
#                     #+++ = 1
#                     octant.iloc[i]=1
#             else:
#                 #++-=5
#                 octant.iloc[i]=5
#         else:
#             if BZ_body.iloc[i]>0:
#                     #+-+ = 4
#                     octant.iloc[i]=4
#             else:
#                 #+-- = 7
#                 octant.iloc[i]=7
#     else:
#         if BY_body.iloc[i]>0:
#             if BZ_body.iloc[i]>0:
#                     #-++ = 2
#                     octant.iloc[i]=2
#             else:
#                 #-+-= 6
#                 octant.iloc[i]=6
#         else:
#             if BZ_body.iloc[i]>0:
#                     #--+ = 3
#                     octant.iloc[i]=3
#             else:
#                 #--- = 8
#                 octant.iloc[i]=8
