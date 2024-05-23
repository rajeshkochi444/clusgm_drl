# %%
import glob
from ase.io import read, write, Trajectory
import matplotlib.pyplot as plt
import seaborn as sns
from asap3 import EMT
import numpy as np
import os
import multiprocessing
from ase import Atoms, neighborlist
from scipy import sparse
import shutil

# %%
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

    return similar

# %%
def checkBonded(clus):
    """
    Check if every atom of the cluster is bonded to other
    """
    cutOff = neighborlist.natural_cutoffs(clus, mult=1)
    neighborList = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
    neighborList.update(clus)
    matrix = neighborList.get_connectivity_matrix(sparse=False)
    n_components, component_list = sparse.csgraph.connected_components(matrix)
    if n_components == 1:
        bonded = True
    else:
        bonded = False
    return bonded

# %%
def generate_unique_clus_list(traj, unique_min_list, similar_min_list, nonbonded_list):
    try:
        for i, img in enumerate(traj):
            if checkBonded(img) == True:
                if len(unique_min_list) == 0:
                    unique_min_list.append(img)
                else:
                    bool_list = []
                    for clus in unique_min_list:
                        similar  = checkSimilar(img, clus)
                        bool_list.append(similar) 
            
                    if any(bool_list):
                        similar_min_list.append(img)
                    else:
                        unique_min_list.append(img)
            else:
                nonbonded_list.append(img)
    except:
        pass
            
    return unique_min_list, similar_min_list, nonbonded_list

# %%
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
        #print(f"Renamed '{file}' to '{new_file_name}'")
    return 

# %%
def process_traj_pair(traj_file1, traj_file2,round_number):
    
    traj_path1 = 'round_' + str(round_number) + '/' + traj_file1
    traj_path2 = 'round_' + str(round_number) + '/' + traj_file2

    unique_min_list = []
    similar_min_list = []
    nonbonded_list = []
    try:
        traj1 = Trajectory(traj_path1)
        unique_min_list, similar_min_list, nonbonded_list = generate_unique_clus_list(traj1, unique_min_list, similar_min_list, nonbonded_list)
    except:
        pass
    
    try:
        traj2 = Trajectory(traj_path2)
        unique_min_list, similar_min_list, nonbonded_list = generate_unique_clus_list(traj2, unique_min_list, similar_min_list, nonbonded_list)  
    except:
        pass
    
    if len(unique_min_list) > 0 :
        save_dir = 'round_' + str(round_number+1) + '/'
        base_name1, file_extension = os.path.splitext(traj_file1)
        base_name2, file_extension = os.path.splitext(traj_file2)
        fname_save = save_dir + base_name1 + '_' +  base_name2 + '.traj'
        write(fname_save, unique_min_list, format='traj' )
    
    return len(unique_min_list), len(similar_min_list), len(nonbonded_list)
   

# %%
def process_traj_second(traj_file1, traj_file2, round_number):
    
    traj_path1 = 'round_' + str(round_number) + '/' + traj_file1
    traj_path2 = 'round_' + str(round_number) + '/' + traj_file2
   
    unique_min_list = []
    similar_min_list = []
    nonbonded_list = []
    try:
        traj1 = Trajectory(traj_path1)
        for img in traj1:
            unique_min_list.append(img)
    except:
        pass

    try:
        traj2 = Trajectory(traj_path2)
        unique_min_list, similar_min_list, nonbonded_list = generate_unique_clus_list(traj2, unique_min_list, similar_min_list, nonbonded_list)
    except:
        pass
    
    if len(unique_min_list) > 0 :
        save_dir = 'round_' + str(round_number+1) + '/'
        base_name1, file_extension = os.path.splitext(traj_file1)
        base_name2, file_extension = os.path.splitext(traj_file2)
        fname_save = save_dir + base_name1 + '_' +  base_name2 + '.traj'
        write(fname_save, unique_min_list, format='traj' )
    
    return len(unique_min_list), len(similar_min_list), len(nonbonded_list)

# %%
def process_single_traj(traj_file, round_number):
    
    traj_path = 'round_' + str(round_number) + '/' + traj_file
    
    unique_min_list = []
    similar_min_list = []
    nonbonded_list = []

    try:
        traj = Trajectory(traj_path)
        if round_number == 0:
            unique_min_list, similar_min_list, nonbonded_list = generate_unique_clus_list(traj, unique_min_list, similar_min_list, nonbonded_list)
        else:
            unique_min_list = [img for img in traj]
    except:
        pass

    if len(unique_min_list) > 0 :
        save_dir = 'round_' + str(round_number+1) + '/'
        base_name1, file_extension = os.path.splitext(traj_file)
        fname_save = save_dir + base_name1 + '_' +  base_name1 + '.traj'
        write(fname_save, unique_min_list, format='traj' )
    
    return len(unique_min_list), len(similar_min_list), len(nonbonded_list)

# %%
def generate_traj_pair(parent_dir):
    files = os.listdir(parent_dir)
    files = [f for f in files if os.path.isfile(os.path.join(parent_dir, f)) and not f.startswith('.')] # Filter out directories and hidden files
    files.sort()
    tot_files = len(files)  # Number of files (N)
    
    #print(files)
    print("Total Files:", tot_files)
    if tot_files % 2 == 0:
        N = tot_files
        single_traj = None
        
    else:
        N = tot_files -1
        single_traj = str(N) + '.traj'
        
    traj_pair = [(f"{i}.traj", f"{i+1}.traj") for i in range(0, N, 2)]
    return traj_pair, single_traj

# %%
def parallel_process(traj_pair, round_number):


    arg_list = [(item[0], item[1], round_number) for item in traj_pair]

    if round_number == 0:
        with multiprocessing.Pool() as pool:
            results = pool.starmap(process_traj_pair, arg_list)
    else:
        with multiprocessing.Pool() as pool:
            results = pool.starmap(process_traj_second, arg_list)
            
    return results

# %%
def has_only_one_traj_file(folder_path):
    # Get a list of files in the directory
    files = os.listdir(folder_path)

    # Count .txt files
    txt_file_count = sum(file.endswith('.traj') for file in files)

    # Check if there is exactly one .txt file
    return txt_file_count == 1


# %%
def write_results(results):
    column_width = 10
    with open('results_unique_traj.txt', 'a+') as file:
        for items in results:
            line = ''.join(f"{str(item):<{column_width}}" for item in items)
            file.write(line + '\n')

# %%
def main():
    round_number = 0
    while not has_only_one_traj_file('round_'+str(round_number)):
        
        parent_dir = 'round_' + str(round_number) + '/' 
        rename_files_in_directory(parent_dir)

        save_dir = 'round_' + str(round_number+1) + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        traj_pair, single_traj = generate_traj_pair(parent_dir)
        #print(traj_pair)
        print(round_number, len(traj_pair), single_traj)
    
        results = parallel_process(traj_pair, round_number)
        #print(results)

        if single_traj is not None:
            results_single_traj = process_single_traj(single_traj, round_number)
            #print(results_single_traj)
            results.append(results_single_traj)
        
        #print(results, '\n')

        new_result_data = []
        for result in results:
            tot_sum = sum(result)
            new_result = (round_number,) + result + (tot_sum,)
            new_result_data.append(new_result)
        write_results(new_result_data)
        
        round_number += 1
        shutil.rmtree(parent_dir)


if __name__ == "__main__":
    main()



