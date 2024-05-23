import glob
from generate_descriptors_amptorch import Generate_acsf_descriptor, Generate_soap_descriptor
from ase.io import read, write, Trajectory
import matplotlib.pyplot as plt
import seaborn as sns
from asap3 import EMT
import numpy as np
#from utils import *

def checkSimilar(clus1, clus2):

    """Check whether two clusters are similar or not by comparing their moments of inertia"""
    Inertia1 = clus1.get_moments_of_inertia()
    Inertia2 = clus2.get_moments_of_inertia()
    # print(Inertia1, Inertia2, 'diff: ', Inertia1-Inertia2)

    tol = 0.01
    if (
        Inertia1[0] * (1 - tol) <= Inertia2[0] <= Inertia1[0] * (1 + tol)
        and Inertia1[1] * (1 - tol) <= Inertia2[1] <= Inertia1[1] * (1 + tol)
        and Inertia1[2] * (1 - tol) <= Inertia2[2] <= Inertia1[2] * (1 + tol)
    ):
        similar = True
    else:
        similar = False

    return similar, Inertia1, Inertia2


img_list = []
ene_list = []
unique_min_list = []
for i, file in enumerate(glob.glob('result_clusgym_ver22_expt1/episode_min/*.traj')):
    if i % 1000 == 0:
        print(i)
    #if i==30:
        #break
    #print(file)
    traj = Trajectory(file)
    #print(traj, len(traj))
    for j, img in enumerate(traj):
        #img.calc = EMT()
        #ene = img.get_potential_energy()
        img_list.append(img)
        if i == 0 and j == 0:
            unique_min_list.append(img)
            print('first min', i, j, len(unique_min_list))
        else:
            bool_list = []
            for clus in unique_min_list:
                similar, inertia1, inertia2 = checkSimilar(img, clus)
                bool_list.append(similar) 
                #print(i,j, bool_list, inertia1, inertia2)
        
            if any(bool_list):
                print(i, j, 'similar minima', len(img_list))
            else:
                unique_min_list.append(img)
                print(i, j, 'new unique_min_found', len(img_list), len(unique_min_list))

write('unique_min_analysis.traj', unique_min_list, format='traj' )
