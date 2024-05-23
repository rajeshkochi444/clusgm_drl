import glob
from ase.io import read, write, Trajectory
import matplotlib.pyplot as plt
import seaborn as sns
#from asap3 import EMT
import numpy as np
import os
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

def rename_files_in_directory(directory):
    # Get all file names in the directory
    files = os.listdir(directory)

    # Filter out directories and hidden files
    files = [f for f in files if os.path.isfile(os.path.join(directory, f)) and not f.startswith('.')]

    # Sort files for consistent ordering
    #files.sort()
    #print(files)

    # Rename each file
    for index, file in enumerate(files):
        # Split the file name and extension
        file_name, file_extension = os.path.splitext(file)
        
        # Create the new file name
        #new_file_name = f"{index}_{file_name}{file_extension}"
        new_file_name = f"{index}{file_extension}"
        
        # Create the full path for the old and new file names
        old_file_path = os.path.join(directory, file)
        new_file_path = os.path.join(directory, new_file_name)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{file}' to '{new_file_name}'")
    return 

def unique_traj_analysis(traj_directory, split_range, round_number):

    files = os.listdir(traj_directory)

    # Filter out directories and hidden files
    files = [f for f in files if os.path.isfile(os.path.join(traj_directory, f)) and not f.startswith('.')]

    tot_files =  len(files)
    q = tot_files // split_range
    

    if tot_files % split_range == 0:
        print(tot_files, q, q*split_range, (q+1)*split_range)
        q = tot_files // split_range
        start_idx_list = list(range(0, (q)*split_range, split_range))
        end_idx_list = list(range(split_range, (q+1)*split_range, split_range))
        print(start_idx_list)
        print(end_idx_list)

    elif tot_files % split_range != 0:
        print(tot_files, q, (q+1)*split_range, (q+2)*split_range)
        q = tot_files // split_range
        start_idx_list = list(range(0, (q+1)*split_range, split_range))
        end_idx_list = list(range(split_range, (q+2)*split_range, split_range))
        print(start_idx_list)
        print(end_idx_list)
    else:
        print("Some error in indexing traj files")    

    for idx, (start_idx, end_idx)  in enumerate(zip(start_idx_list, end_idx_list)):
        print('current idx:', idx, start_idx, end_idx)

        img_list = []
        unique_min_list = []
        similar_min_list = []

        for  kk in range(start_idx,end_idx):
            traj_file = traj_directory + str(kk) + '.traj'
            print(kk, traj_file)
            try:
                traj = Trajectory(traj_file)

                for j, img in enumerate(traj):
                    img_list.append(img)
                
                    if kk == start_idx and j == 0:
                        unique_min_list.append(img)
                        print('first min', kk, j, len(unique_min_list))
                    else:
                        bool_list = []
                        for clus in unique_min_list:
                            similar, inertia1, inertia2 = checkSimilar(img, clus)
                            bool_list.append(similar) 
                
                        if any(bool_list):
                            similar_min_list.append(img)
                            print(kk, j, 'similar minima', len(img_list), len(similar_min_list))
                        else:
                            unique_min_list.append(img)
                            print(kk, j, 'new unique_min_found', len(img_list), len(similar_min_list), len(unique_min_list))
            except:
                print(f'{traj_file} not found')
            print('\n')

        print(f"writing trajectory after each step: idx: {idx} start_idx {start_idx} end_idx {end_idx} round_number {round_number}")

        save_dir = 'Analysis_Unique_Traj/' + "round_" + str(round_number) + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fname_save = save_dir + "round_" + str(round_number)+ '_' + str(idx) + '_' +  str(start_idx)+ '_' + str(end_idx) + '_' + str(len(unique_min_list)) + '.traj'
        write(fname_save, unique_min_list, format='traj' )
        print(fname_save, len(unique_min_list))
        print('\n')      
    return 

def main():

    if not os.path.exists('Analysis_Unique_Traj/'):
        os.makedirs('Analysis_Unique_Traj/')
    round_number  = 0
    split_range = 5
    traj_directory = 'episode_min_test/'

    files = os.listdir(traj_directory)
    # Filter out directories and hidden files
    files = [f for f in files if os.path.isfile(os.path.join(traj_directory, f)) and not f.startswith('.')]
    tot_files =  len(files)

    print(round_number, traj_directory, tot_files)

    unique_traj_analysis(traj_directory, split_range, round_number)

    for i in range(1000):
        round_number = i
        split_range = 2
        traj_directory = 'Analysis_Unique_Traj/' + "round_" + str(round_number) + '/'
        rename_files_in_directory(traj_directory)

        files = os.listdir(traj_directory)

        # Filter out directories and hidden files
        files = [f for f in files if os.path.isfile(os.path.join(traj_directory, f)) and not f.startswith('.')]

        tot_files =  len(files)

        print(round_number, traj_directory, tot_files)
        if tot_files == 1:
            break
        else:
            unique_traj_analysis(traj_directory, split_range, round_number+1)

if __name__ == '__main__':
    main()
