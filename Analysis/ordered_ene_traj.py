import ase
import os
import glob
#from ase.io.trajectory import Trajectory
import numpy as np
from ase.io import read, write, Trajectory
from ase.calculators.emt import EMT
from ase.io.trajectory import Trajectory, TrajectoryReader, TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator as sp

def ordered_ene_traj(traj_file, desc=False):
    traj = Trajectory(traj_file)
    print("Total number of images in trajectory:", len(traj))
    img_list = []
    ene_list = []
    for i, img in enumerate(traj):
        img.calc = EMT()
        ene = img.get_potential_energy()
        #print(i, ene)
        ene_list.append(ene)
        img_list.append(img)

    #print(len(ene_list), len(img_list))

    values = np.array(ene_list)

    # Get the indices that would sort the ene values array
    sorted_indices_asc = np.argsort(values)  # For ascending order
    print(sorted_indices_asc)
    # Use these indices to order the traj file names
    img_list_ordered_asc = [img_list[i] for i in sorted_indices_asc]
    write('asc_'+ traj_file, img_list_ordered_asc)

    if desc == True:
        sorted_indices_desc = np.argsort(-values)  # For descending order
        print(sorted_indices_desc)
        img_list_ordered_desc = [img_list[i] for i in sorted_indices_desc]
        write(traj_file + '_des.traj', img_list_ordered_desc)
    return None

def write_cif(traj_file, folder_name, n_images):
    os.makedirs(folder_name, exist_ok=True)
    traj = Trajectory(traj_file)
    for i in range(n_images):
        img = traj[i]
        print(img)
        cif_fname = folder_name + '/' + str(i) + '.cif'
        write(cif_fname, img, format='cif')


def print_energies(traj_file,n_images):
    traj = Trajectory(traj_file)
    for i, img in enumerate(traj):
        if i == n_images:
            break
        img.calc = EMT()
        print(img.get_potential_energy())

ordered_ene_traj("0_1.traj")
print_energies("asc_0_1.traj", 25)
write_cif("asc_0_1.traj", 'Lowest_Configs', 10)
