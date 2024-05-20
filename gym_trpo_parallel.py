import matplotlib
matplotlib.use("Agg")
import gym
#from surface_seg.envs.mcs_env import MCSEnv
from clusgym  import MCSEnv
import gym.wrappers
import numpy as np
import tensorforce 
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.execution import Runner
import os
import copy
from callback import Callback

timesteps = 200
num_parallel = 24
#seed = 30
eleNames = ['Cu', 'Ni', 'Au', 'Pd']
eleNums = [ 4,5,6,5]
clus_seed = None
save_dir =  'result_' + ''.join(f"{name}{num}" for name, num in zip(eleNames, eleNums)) + '/'



def setup_env(recording=False):
    
    # Set up gym
    #MCS_gym = MCSEnv(fingerprints=True, 
                    #permute_seed=None)
   
    # Set up gym
    MCS_gym = MCSEnv(eleNames=eleNames,
                     eleNums=eleNums,
                     clus_seed=clus_seed,
                     observation_fingerprints=True,
                     save_dir = save_dir,
                     timesteps = timesteps,
                     save_every = 1,
                     n_unique_pool = 25, 
                    )


 
    #if recording:
    # Wrap the gym to provide video rendering every 50 steps
        #MCS_gym = gym.wrappers.Monitor(MCS_gym, 
                                         #"./vid", 
                                         #force=True,
                                        #video_callable = lambda episode_id: (episode_id)%50==0) #every 50, starting at 51
    
    #Convert gym to tensorforce environment
    env = tensorforce.environments.OpenAIGym(MCS_gym,
                                         max_episode_timesteps=400,
                                         visualize=False)
    
    return env

"""
Create a environment for checking the intial energy and thermal energy
"""
#env = setup_env().environment.env
#print('initial energy', env.initial_energy)

agent = Agent.create(
    agent='trpo', 
    environment=setup_env(), 
    batch_size=10, 
    learning_rate=1e-2,
    memory = 40000,
    max_episode_timesteps = 400,
    parallel_interactions = num_parallel,
    exploration=dict(
        type='decaying', unit='timesteps', decay='exponential',
        initial_value=0.3, decay_steps=1000, decay_rate=0.5
    ))

# Check agent specifications
agent_spec = agent.spec
print(agent_spec)

#num_processes = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])

#print('Detected N=%d cores, running in parallel!'%num_processes)

#plot_frequency --> plotting energy and trajectories frequency
callback = Callback(save_dir).episode_finish

num_processes = num_parallel
runner = Runner(
    agent=agent,
    environments=[setup_env() for _ in range(num_processes)],
    num_parallel=num_processes,
    remote='multiprocessing',
    max_episode_timesteps=timesteps,
)

runner.run(num_episodes=64000,  callback=callback, callback_episode_frequency=1)
#runner.run(num_episodes=100, evaluation=True)
runner.close()
