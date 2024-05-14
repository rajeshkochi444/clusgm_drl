# clusgm_drl
### DRL Framework for Nanocluster Global Minimum Search 
We have developed a pioneering Deep Reinforcement Learning (DRL) framework for exploring the potential energy surfaces of nanoclusters, efficiently identifying ground state and low-energy configurations across various cluster types and sizes. This innovation represents the first application of DRL in the ground state search for nanoclusters, demonstrating exceptional adaptability and potential for advancing materials science.
![image](https://github.com/rajeshkochi444/clusgm_drl/assets/40799655/0547a99e-ad53-4427-be5d-3a10084148a3)

![image](https://github.com/rajeshkochi444/clusgm_drl/assets/40799655/073b9e61-9ab2-4308-8e0c-f646a9a4f5de)

### How to Run the Code

1. **Set Up the Environment:**
   - Install the required Conda environment from the provided YAML file:
     ```bash
     conda env create -f env_clusgym.yml
     ```

2. **Configure the Nanocluster Composition:**
   - Edit the `gym_trpo_parallel.py` or `gym_trpo_single.py` file to select the desired nanocluster composition. For example, to simulate a cluster of 20 copper (Cu) atoms:
     ```python
     eleNames = ['Cu']
     eleNums = [20]
     ```
   Example for mutimetallic nanocluster compositions
    ```python
     eleNames = ['Cu', 'Ag']
     eleNums = [20, 23]
     ```
4. **Run the Simulation:**
   - Execute the script using Python. You can run either the parallel or single version depending on your needs:
     ```bash
     python gym_trpo_parallel.py  # For parallel execution
     python gym_trpo_single.py    # For single execution
     ```

5. **Utilize the SLURM Script:**
   - An example SLURM script is provided for running the simulations on clusters. Modify the script as necessary to match your cluster configuration:
     ```bash
     sbatch slurm_script.sh
     ```

These steps will guide you through running the simulations using the Deep Reinforcement Learning framework for nanoclusters.
