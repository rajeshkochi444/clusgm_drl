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
num_parallel = 48
#seed = 30
eleNames = ['Cu']
eleNums = [20]
clus_seed = None
save_dir = 'result_' + ''.join(f"{name}{num}" for name, num in zip(eleNames, eleNums)) + '/'


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

env = setup_env().environment

#from ase.io import write
#write("initial_clus.traj", env.atoms)

agent = Agent.create(
    agent='trpo', 
    environment=setup_env(), 
    batch_size=10, 
    learning_rate=1e-2,
    memory = 40000,
    max_episode_timesteps = 400,
    exploration=dict(
        type='decaying', unit='timesteps', decay='exponential',
        initial_value=0.3, decay_steps=1000, decay_rate=0.5
    ))

agent_spec = agent.spec
print(agent_spec)
#plot_frequency --> plotting energy and trajectories frequency

callback = Callback(save_dir).episode_finish

runner2 = Runner(
    agent=agent,
    environment=setup_env(recording=False),
    max_episode_timesteps=timesteps,
)

# %prun runner.run(num_episodes=2, callback=callback, callback_episode_frequency=1)

# callback_episode_frequency --> saving results and trajs frequency
runner2.run(num_episodes=32000, callback=callback, callback_episode_frequency=1)
# runner2.run(num_episodes=100, evaluation=True)
runner2.close()
