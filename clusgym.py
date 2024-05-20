import gym
from gym import spaces
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.optimize.bfgs import BFGS
from ase.visualize.plot import plot_atoms
from ase.io import write
from asap3 import EMT
import copy
import random
from utils import *
import itertools
from generate_descriptors import Generate_acsf_descriptor, Generate_soap_descriptor
from mutations import do_nothing, homotop, rattle_mut, rotate_mut, twist, partialInversion, tunnel, skin, changeCore, mate


DIRECTION =[+2.0, -2.0 ]


class MCSEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, 
                 eleNames=None,
                 eleNums=None,
                 clus_seed=None,
                 save_dir=None,
                 observation_fingerprints = True,
                 observation_forces=True,
                 observation_positions = True,
                 descriptors = None,
                 timesteps = None,
                 save_every = None,
                 plot_every = None,
                 n_unique_pool = None,
                 
                ):
        
        self.eleNames = eleNames
        self.eleNums  = eleNums
        self.eleRadii = [covalent_radii[atomic_numbers[ele]] for ele in self.eleNames]
        self.avg_radii = sum(self.eleRadii) / len(self.eleNums)
        self.clus_seed = clus_seed
        self.descriptors = descriptors
        self.timesteps = timesteps
        self.save_every = save_every
        self.plot_every = plot_every
        self.save_dir = save_dir
        self.n_unique_pool = n_unique_pool
        self.episodes = 0
        self.counter = 0

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.history_dir = os.path.join(save_dir, 'history')
        self.plot_dir = os.path.join(save_dir, 'plots')
        self.traj_dir = os.path.join(save_dir, 'trajs')
        self.episode_min_traj_dir = os.path.join(save_dir, 'episode_min')
        self.unique_min_traj_dir = os.path.join(save_dir, 'unique_min')
        self.episode_lowest_min_traj_dir = os.path.join(save_dir, 'episode_lowest_min')

        for folder in [self.history_dir, self.plot_dir, self.traj_dir, self.episode_min_traj_dir, self.unique_min_traj_dir, self.episode_lowest_min_traj_dir]:
            if not os.path.exists(folder):
                os.makedirs(folder)
              
        self.initial_atoms, self.elements = self._get_initial_clus()
        self.initial_atoms.set_calculator(EMT())
        self.initial_positions = self.initial_atoms.get_positions()

        self.new_initial_atoms = self.initial_atoms.copy()
        self.initial_energy = self.initial_atoms.get_potential_energy()
        
        self.relative_energy = 0.0
        self.initial_forces = self.initial_atoms.get_forces()
        
        self.atoms = self.initial_atoms.copy()
        self.clus_size = len(self.atoms)

        self.observation_positions = observation_positions
        self.observation_fingerprints = observation_fingerprints
        self.observation_forces = observation_forces

        #self.observation_nlow_min = True
        #self.observation_nhigh_min = True


        self.fps, self.fp_length = self._get_fingerprints(self.atoms)
        self.episode_initial_fps = self.fps.copy()
        self.positions = self.atoms.get_positions()
        
        self.reach_convergence = 0
        self.n_tot_all_minima = 0
        self.n_lower_energy_minima = 0
        self.n_higher_energy_minima = 0
        self.n_high_low_minima = 0
        self.n_overlap = 0
        self.n_nonbonded = 0
        self.n_similar_min = 0
        self.total_episode_steps = 0

        self.found_overlap = 0
        self.found_nonbonded = 0
        self.found_low_min = 0
        self.found_high_min = 0
        self.found_similar_min = 0
        

        #unique minima 
        #self.unique_minima = [self.initial_atoms.copy()]
        #self.unique_minima_energies = [0.0]
        self.unique_minima = []
        self.unique_minima_energies = []
        self.n_unique_minima = 0


        # Define the possible actions

        self.action_space = spaces.Dict({'atom_selection': spaces.Discrete(self.clus_size),
                                         'movement':spaces.Discrete(2) } ) 
        #Define the observation space
        self.observation_space = self._get_observation_space()
        
        # Set up the initial atoms
        self.reset()
        
        return

      # open AI gym API requirements
    def step(self, action):
        '''
            The agent will perform an action based on the atom selection and movement. 
            The selected atom's coordinate will be shifted from the current position.
            if the movement from the current position results in overlapped atoms,
            the reward wil be -1000
            if there is no overlapping, geometry relaxation will be performed.
            if it found  a new minimum configuration, the step will get reward based on the 
            relative energy from the initial minimum.
            if it relaxed to an existing minimum found earlier in the episode,
            the reward will be zero.
        '''
        
        reward = 0  

        self.atom_selection = action['atom_selection']
        self.movement = action['movement']
        self.shift = DIRECTION[self.movement]
       
        self.done= False
        episode_over = False

        save_path_min = None



    	#shifting the position of the selected atom in the cluster
        #self.atoms[self.atom_selection].position = self.atoms.get_positions()[self.atom_selection] + self.shift * 2.5
        self.atoms = self._clus_move_atom(self.atoms, self.atom_selection, self.shift)

        dist_after_move = self.atoms.get_all_distances()[self.atom_selection]
        z1 = [ dist for k, dist in enumerate(dist_after_move) if k != self.atom_selection and dist < 0.5*self.avg_radii ]
	
        if len(z1) > 0:     	#checking for overlapping atoms after movement
            reward -= 10.0
            self.n_overlap += 1
            self.found_overlap = 1

        if checkBonded(self.atoms) == False:
            reward -= 10.0
            self.n_nonbonded += 1
            self.found_nonbonded = 1

        if not ( len(z1) > 0 or checkBonded(self.atoms) == False):			#minimization
            dyn = BFGS(atoms=self.atoms, logfile=None, trajectory= save_path_min)
            #converged = dyn.run(fmax=0.02)
            dyn.run(fmax=0.02)
            self.relative_energy = self._get_relative_energy()

            self.all_minima['minima'].append(self.atoms.copy())
            self.all_minima['energies'].append(self._get_relative_energy())
            self.all_minima['timesteps'].append(self.history['timesteps'][-1] + 1)
            self.all_minima['positions'].append(self.atoms.positions.copy())

            #self.n_tot_all_minima = len(self.all_minima['minima']) - 1 #initial clus added to the list. so removing it from the count
            self.n_tot_all_minima += 1
            #reward += 2**(self.n_tot_all_minima)


		
		    #checking the similarilty of the relaxed cluster minimum  between cluster minima already found
            bool_list = []
            for clus in self.minima['minima']:
                bool_list.append(checkSimilar(self.atoms, clus)) 
		
            if any(bool_list): #if the minimum is already found, reward is -100.0
                reward -= 10.0
                self.n_similar_min += 1
                self.found_similar_min =  1
            else:				# a new minima found
                if self.relative_energy < 0.0:
                    #reward += 100 * np.exp((-1.0) * self.relative_energy)
                    reward += 1000 * abs(self.relative_energy)
                    self.n_lower_energy_minima += 1
                    self.found_low_min = 1
                else:
                    reward += 0.0
                    #reward +=  100 * np.exp((+1.0) * self.relative_energy)
                    #reward += 1000 * abs(self.relative_energy)
                    self.n_higher_energy_minima += 1
                    self.found_high_min = 1
                
                self.n_high_low_minima = self.n_lower_energy_minima + self.n_higher_energy_minima
		
                self.minima['minima'].append(self.atoms.copy())
                self.minima['energies'].append(self._get_relative_energy())
                self.minima['timesteps'].append(self.history['timesteps'][-1] + 1)
                self.minima['positions'].append(self.atoms.positions.copy())

                #checking and adding whether the relaxed cluster is a  new unique minimum found.  
                #unique minimum can provide all the unique minima that were found from different episodes. 
                unique_bool_list = []
                for clus in self.unique_minima:
                    unique_bool_list.append(checkSimilar(self.atoms, clus)) 
            
                if any(unique_bool_list): #if the minimum is already found in unique minima list, reward is zero
                    reward += 0.0
                else:				# a new unique minima found	
                    self.unique_minima.append(self.atoms.copy())
                    self.unique_minima_energies.append(self._get_absolute_energy())
            
            if self.n_lower_energy_minima == 5:
            #if self.n_high_low_minima == 10:
            #if self.n_tot_all_minima == 10:
                self.done = True
                self.reach_convergence = 1


                
        #Fingerprints after step action
        self.fps, self.fp_length = self._get_fingerprints(self.atoms)
       
        #Get the new observation
        observation = self._get_observation()
        

        #Update the history for the rendering after each step
        self.relative_energy = self._get_relative_energy()
       	
        self.trajectories.append(self.atoms.copy())

        self.history['timesteps'] = self.history['timesteps'] + [self.history['timesteps'][-1] + 1]
        self.history['energies'] = self.history['energies'] + [self.relative_energy]
        self.history['positions'] = self.history['positions'] + [self.atoms.get_positions(wrap=False).tolist()]
        self.history['scaled_positions'] = self.history['scaled_positions'] + [self.atoms.get_scaled_positions(wrap=False).tolist()]
        if self.observation_fingerprints:
            self.history['fingerprints'] = self.history['fingerprints'] + [self.fps.tolist()]
            self.history['initial_fps'] = self.history['initial_fps'] + [self.episode_initial_fps.tolist()]

        self.episode_reward += reward
        self.total_episode_steps += 1

        #if len(self.history['actions'])-1 >= self.total_steps:
        if self.done:
            episode_over = True
        elif len(self.history['timesteps'])-1 >= self.total_steps:
            episode_over = True
            
        if episode_over: 
            self.min_idx = int(np.argmin(self.minima['energies']))
            self.unique_min_idx = int(np.argmin(self.unique_minima_energies))
            self.episode_lowest_min = self.minima['minima'][self.min_idx]
            #print("Total clus, Total unique clus:", len(self.minima['minima']), len(self.unique_minima))
            if self.episodes % self.save_every == 0:
                self.save_episode()
                self.save_traj()

            # Updaing the unique minima 
            self.n_unique_minima_before = len(self.unique_minima) # After adding all unique minimum configuations in each episode
            
            if self.episodes % self.save_every == 0:
                ene_index_list = np.argsort(self.unique_minima_energies)
                if self.n_unique_minima >= self.n_unique_pool:
                    self.unique_minima = [self.unique_minima[i] for i in ene_index_list[:self.n_unique_pool]]
                    self.unique_minima_energies = [self.unique_minima_energies[i] for i in ene_index_list[:self.n_unique_pool]]
                else:
                    self.unique_minima = [self.unique_minima[i] for i in ene_index_list]
                    self.unique_minima_energies = [self.unique_minima_energies[i] for i in ene_index_list]
                unique_low_images_fname = self.save_dir + 'unique_low_images.traj'
                write(unique_low_images_fname, self.unique_minima, format='traj' )
            self.n_unique_minima = len(self.unique_minima) #should be 10 maximum as we are taking only 10 low geoms

            plt.plot(self.unique_minima_energies)
            plt.savefig(self.save_dir + 'unique_min.png')
            plt.close()
                    
                
            self.episodes += 1
            with open(self.save_dir + 'episode_data.txt', "a+") as fh:
                fh.write(f"Ep_number: {self.episodes}, "
                         f"Ep_Reward: {self.episode_reward: .1f}, "
                         f"Ep_tot_steps: {self.total_episode_steps}, "
                         f"T_overlap: {self.n_overlap}, "
                         f"T_nonbonded: {self.n_nonbonded}, "
                         f"T_tot_all_min: {self.n_tot_all_minima}, "
                         f"T_similar_min: {self.n_similar_min}, "
                         f"T_lower_ene_min: {self.n_lower_energy_minima}, "
                         f"T_higher_ene_min: {self.n_higher_energy_minima}, "
                         f"T_high_low_min: {self.n_high_low_minima}, "
                         f"T_unique_min: {self.n_unique_minima_before, self.n_unique_minima}, "
                         f"Initial Ene: {(self.initial_energy): .4f}, "
                         f"GM_ene: {min(self.unique_minima_energies): .4f}, "
                         f"atom select: {self.atom_selection}, "
                         f"atom_shift: {self.shift} \n " 
                         )
            
        return observation, reward, episode_over, {}

    def _get_initial_clus(self):
        self.initial_atoms, self.elements = self._generate_clus(self.counter)   
        return self.initial_atoms, self.elements
    
    def save_episode(self):
        save_path = os.path.join(self.history_dir, '%d_%f_%f_%f_%d_%d_%d.npz' %(self.episodes, self.minima['energies'][self.min_idx],
                                                                   self.initial_energy, self.episode_reward, self.n_tot_all_minima, self.n_lower_energy_minima, self.n_unique_minima))
        np.savez_compressed(save_path, 
             initial_energy = self.initial_energy,
             energies = self.history['energies'],
             #actions = self.history['actions'],
             scaled_positions = self.history['scaled_positions'],
             fingerprints = self.history['fingerprints'],
             initial_fps = self.history['initial_fps'],
             minima_energies = self.minima['energies'],
             minima_steps = self.minima['timesteps'],
             reward = self.episode_reward,
             episode = self.episodes,
            )
        return
    
    def save_traj(self):      
        save_path = os.path.join(self.traj_dir, '%d_%f_%f_%f_%d_%d_%d_full.traj' %(self.episodes, self.minima['energies'][self.min_idx]
                                                                      , self.initial_energy, self.episode_reward, self.n_tot_all_minima, self.n_lower_energy_minima, self.n_unique_minima))
        episode_min_path = os.path.join(self.episode_min_traj_dir, '%d_%f_%f_%f_%d_%d_%d_full.traj' %(self.episodes, self.minima['energies'][self.min_idx]
                                                                      , self.initial_energy, self.episode_reward, self.n_tot_all_minima, self.n_lower_energy_minima, self.n_unique_minima))
        unique_min_path = os.path.join(self.unique_min_traj_dir, '%d_%f_%f_%f_%d_%d_%d_full.traj' %(self.episodes, self.unique_minima_energies[self.min_idx]
                                                                      , self.initial_energy, self.episode_reward, self.n_tot_all_minima, self.n_lower_energy_minima, self.n_unique_minima))
        episode_lowest_min_path = os.path.join(self.episode_lowest_min_traj_dir, '%d_%d_%f_%f_%f_full.traj' %(self.episodes, self.min_idx, self.minima['energies'][self.min_idx]
                                                                      , self.initial_energy, self.episode_reward))
        trajectories = []
        for atoms in self.trajectories:
            atoms.set_calculator(EMT())
            trajectories.append(atoms)
        write(save_path, trajectories)

        write(episode_min_path, self.minima['minima'])
        write(unique_min_path, self.unique_minima)
        write(episode_lowest_min_path, self.episode_lowest_min)
        return

    def reset(self):
        #Copy the initial atom and reset the calculator
            
        self.new_initial_atoms, self.elements = self._get_initial_clus()
        self.atoms = self.new_initial_atoms.copy()
        self.atoms.set_calculator(EMT())
        self.episode_reward = 0
        self.total_steps = self.timesteps
        self.counter = 0
      
        #Reset the list of identified minima and their energies and positions
        self.minima = {}
        self.minima['minima'] = [self.atoms.copy()]
        self.minima['energies'] = [0.0]
        self.minima['positions'] = [self.atoms.positions.copy()]
        self.minima['timesteps'] = [0]
       

        self.all_minima = {}
        self.all_minima['minima'] = [self.atoms.copy()]
        self.all_minima['energies'] = [0.0]
        self.all_minima['positions'] = [self.atoms.positions.copy()]
        self.all_minima['timesteps'] = [0]

        self.reach_convergence = 0
        self.n_tot_all_minima = 0
        self.n_lower_energy_minima = 0
        self.n_higher_energy_minima = 0
        self.n_high_low_minima = 0
        self.n_overlap = 0
        self.n_nonbonded = 0
        self.n_similar_min = 0
        self.total_episode_steps = 0

        self.found_overlap = 0
        self.found_nonbonded = 0
        self.found_low_min = 0
        self.found_high_min = 0
        self.found_similar_min = 0


        # Checking whether the initial cluster passed into the DRL framweork is already existing in the unique list or not 
        unique_bool_list = []
        if len(self.unique_minima) ==0:
            self.unique_minima.append(self.atoms.copy())
            self.unique_minima_energies.append(self._get_absolute_energy())
        else:
            for clus in self.unique_minima:
                unique_bool_list.append(checkSimilar(self.atoms, clus)) 
        
            if any(unique_bool_list): #if the minimum is already found in unique minima list, we will not add to the unique pool list
                pass
            else:				
                self.unique_minima.append(self.atoms.copy())
                self.unique_minima_energies.append(self._get_absolute_energy())
        
        self.n_unique_minima = len(self.unique_minima)

        #Set the energy history
        results = ['timesteps', 'energies', 'positions', 'scaled_positions', 'fingerprints', 'initial_fps']
        self.history = {}
        for item in results:
            self.history[item] = []

        self.history['timesteps'] = [0]
        self.history['energies'] = [0.0]
        self.history['positions'] = [self.atoms.get_positions().tolist()]
        self.history['scaled_positions'] = [self.atoms.get_scaled_positions().tolist()]
        if self.observation_fingerprints:
            self.fps, fp_length = self._get_fingerprints(self.atoms)
            self.initial_fps = self.fps
            self.episode_initial_fps = self.fps
            self.history['fingerprints'] = [self.fps.tolist()]
            self.history['initial_fps'] = [self.episode_initial_fps.tolist()]
        
        self.trajectories = [self.atoms.copy()]        
        
        return self._get_observation()
    

    def render(self, mode='rgb_array'):

        if mode=='rgb_array':
            # return an rgb array representing the picture of the atoms
            
            #Plot the atoms
            fig, ax1 = plt.subplots()
            plot_atoms(self.atoms, 
                       ax1, 
                       rotation='48x,-51y,-144z', 
                       show_unit_cell =0)
            
            ax1.set_ylim([0,25])
            ax1.set_xlim([-2, 20])
            ax1.axis('off')
            ax2 = fig.add_axes([0.35, 0.85, 0.3, 0.1])
            
            #Add a subplot for the energy history overlay           
            ax2.plot(self.history['timesteps'],
                     self.history['energies'])
            
            ax2.plot(self.minima['timesteps'],
                    self.minima['energies'],'o', color='r')
        

            ax2.set_ylabel('Energy [eV]')
            
            #Render the canvas to rgb values for the gym render
            plt.draw()
            renderer = fig.canvas.get_renderer()
            x = renderer.buffer_rgba()
            img_array = np.frombuffer(x, np.uint8).reshape(x.shape)
            plt.close()
            
            #return the rendered array (but not the alpha channel)
            return img_array[:,:,:3]
            
        else:
            return
    
    def close(self):
        return
    
    
    def _get_relative_energy(self):
        return self.atoms.get_potential_energy() - self.initial_energy
    
    def _get_absolute_energy(self):
        return self.atoms.get_potential_energy() 

    def _get_observation(self):
        # helper function to get the current observation, which is just the position
        # of the free atoms as one long vector
           
        observation = {'energy':np.array(self._get_relative_energy()).reshape(1,)}
        
        if self.observation_fingerprints:
            observation['fingerprints'] = (self.fps - self.episode_initial_fps).flatten()
            
        
        observation['positions'] = self.atoms.get_scaled_positions().flatten()
            
        if self.observation_forces:
            observation['forces'] = self.atoms.get_forces().flatten()
        
        observation['reach_convergence'] = self.reach_convergence
        observation['found_nonbonded'] = self.found_nonbonded
        observation['found_overlap'] = self.found_overlap
        observation['found_low_min'] = self.found_low_min
        observation['found_high_min'] = self.found_high_min
        observation['found_similar_min'] = self.found_similar_min
        
        #observation['n_tot_all_min'] = self.n_tot_all_minima
        #observation['n_similar_min'] = self.n_similar_min
        #observation['n_high_low_min'] = self.n_high_low_minima
        #observation['n_low_min'] = self.n_lower_energy_minima
        #observation['n_high_min'] = self.n_higher_energy_minima

        return observation
    
    def _get_fingerprints(self, atoms):
        
        fps  = Generate_acsf_descriptor(self.atoms)
        fp_length = fps.shape[-1]
        #print("self.acsf, self.acsf_length")
        #print(fps, fp_length)

        fp_soap  = Generate_soap_descriptor(self.atoms)
        fp_soap_length = fp_soap.shape[-1]
        #print("self.soap, self.soap")
        #print(fp_soap, fp_soap_length)

        return fps, fp_length
    
    def _get_observation_space(self):  
        
        observation_space = spaces.Dict({'fingerprints': spaces.Box(low=-6,
                                            high=6,
                                            shape=(len(self.atoms)*self.fp_length, )),
                                        'positions': spaces.Box(low=-1,
                                            high=2,
                                            shape=(len(self.atoms)*3,)),
                                        'energy': spaces.Box(low=-1,
                                                    high=2.5,
                                                    shape=(1,)),
                                        'forces': spaces.Box(low= -2,
                                                            high= 2,
                                                            shape=(len(self.atoms)*3,)
                                                            ),
                                        'reach_convergence': spaces.Discrete(2),
                                        'found_nonbonded': spaces.Discrete(2),
                                        'found_overlap': spaces.Discrete(2),
                                        'found_low_min': spaces.Discrete(2),
                                        'found_high_min': spaces.Discrete(2),   
                                        'found_similar_min': spaces.Discrete(2),                                
                                        #'n_tot_all_min': spaces.Discrete(201),                                
                                        #'n_similar_min': spaces.Discrete(201),                                
                                        #'n_high_low_min': spaces.Discrete(201),                                
                                        #'n_low_min': spaces.Discrete(201),                                
                                        #'n_high_min': spaces.Discrete(201),                                
                                        })

        return observation_space

    def _generate_clus(self, counter):
        """
	    Generate a random cluster configuration
	    """
        #if self.clus_seed is None:
            #self.clus_seed = random.randint(1, 100000)
            #np.random.seed(self.clus_seed)

        def random_geom(eleNames, eleRadii, eleNums):
            ele_initial = [eleNames[0], eleNames[-1]]
            d = (eleRadii[0] + eleRadii[-1]) / 2
            clusm = Atoms(ele_initial, [(-d, 0.0, 0.0), (d, 0.0, 0.0)])
            clus = addAtoms(clusm, eleNames, eleNums)
            clus = fixOverlap(clus)
            return clus

        random_selection = random.choice([0,1])
        if random_selection == 0:
            clus = random_geom(self.eleNames, self.eleRadii, self.eleNums)
        else:
            clus1 = random_geom(self.eleNames, self.eleRadii, self.eleNums)
            clus2 = random_geom(self.eleNames, self.eleRadii, self.eleNums)
            clus = mate(clus1, clus2)

        ene_before_mut = clus.get_potential_energy()

        if len(self.eleNames) == 1:
            mut = random.choice([do_nothing, rattle_mut, rotate_mut, twist, partialInversion, tunnel, skin, changeCore] )
            clus = mut(clus)
        else:
            mut = random.choice([do_nothing, homotop, rattle_mut, rotate_mut, twist, partialInversion, tunnel, skin, changeCore] )
            clus = mut(clus)
        
        if checkBonded(clus) == False:
            if not counter == 3:
                counter += 1
                clus, elements = self._generate_clus(counter)

        ene_after_mut = clus.get_potential_energy()

        with open("initial_clus_gen.txt", "a+") as fh:
            fh.write(f"random_selection: {random_selection} counter: {counter} mutation: {mut} ene_before_mut: {ene_before_mut} ene_after_mut: {ene_after_mut}  \n")

        elements = np.array(clus.symbols)
        _, idx = np.unique(elements, return_index=True)
        elements = list(elements[np.sort(idx)])
        
        return clus, elements

    def _clus_move_atom(self, clus, atom_idx, d):
        '''
        Move cluster atoms based on action space selections
        '''
        clus_pos = clus.get_positions()
        p1 = clus.get_center_of_mass() 
        p2 = clus_pos[atom_idx,:]
        v = p2 - p1
        mod_v = np.sqrt(v[0]**2+v[1]**2+v[2]**2)
        unit_vec = v/mod_v

        q1 =  clus_pos[atom_idx,:] + unit_vec * d
        clus_pos[atom_idx,:] = q1

        clus.set_positions(clus_pos)
        return clus
    
    
