import glob
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




start_idx_list = list(range(0, 12500, 500))
end_idx_list = list(range(500, 13000, 500))

for start_idx, end_idx  in zip(start_idx_list, end_idx_list):
    print('current idx:', start_idx, end_idx)

    img_list = []
    ene_list = []
    unique_min_list = []

    for  kk in range(start_idx,end_idx):
        print(kk)
        file_path = '../result_clusgym_ver50_expt1/episode_min/'+ str(kk) + '_*.traj'
        #print(i, file_path)
        try:
            for i, file in enumerate(glob.glob(file_path)):
                #if i % 100 == 0:
                    #print(i)
                traj = Trajectory(file)
                for j, img in enumerate(traj):
                    img_list.append(img)
                    if i == 0 and j == 0:
                        unique_min_list.append(img)
                        print('first min', i, j, len(unique_min_list))
                    else:
                        bool_list = []
                        for clus in unique_min_list:
                            similar, inertia1, inertia2 = checkSimilar(img, clus)
                            bool_list.append(similar) 
                
                        if any(bool_list):
                            print(i, j, 'similar minima', len(img_list))
                        else:
                            unique_min_list.append(img)
                            print(i, j, 'new unique_min_found', len(img_list), len(unique_min_list))
        except:
            print(f'{file_path} not found')

    write('unique_min_analysis_'+ str(start_idx)+ '_' + str(end_idx) + '.traj', unique_min_list, format='traj' )
    print('\n')
