## Policy Gradient for Forward synthesis implementation

####File Layout:
* configs: the directory containing configuration files
* envs: the directory that contains gym.Env files and initializers
* external: Apollo1060 and svgstack source (modified by me) are there
* rl: everything that has to do with the agent is there
    - actor_critic.py: pytorch modules for AC networks
    - agents.py: PGFS_Agent class
    - recorder.py: MultiTrack class, which is used to gather the statistics during the training
    - rlutils.py: miscellaneous
    - runner.py: runner class with agent training loop
* chemwrapped.py: wrapped rdkit functions
* chemutils.py: ChemWorld class, that is responsible for loading and managing the templates/reactnats sets
* datasets.py: low level dataset routines
* forward_model.py/function_set_basic.py: define the forward reaction model for the environment
* scoring_functions.py: wraps Apollo1060 scoring functions and provide scaling/normalizing of the reward
* toy_set_generator.py: botched generator for the toy set (not included)
* utilspy: miscellaneous

###Setting up

envirnment.yml file contains the definitions of the conda environment needed to run the code. 
```
conda env create -f environment.yml
```
Note: the pytorch-gpu will, unfortunately, not get installed this way by default, so the following will have to be run (of course with the appropriate cuda version):
```
conda remove pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```
The home directory '~' must contain the .chemPGFS directory with the raw datasets (provided as chemPGFS in the archive).

###Training an Agent
The agent training can be initiated with the following command:
``` 
python train_PGFS.py "./data" "./gym_PGFS/configs/config_server_default.yaml"
```
The 'data' folder can be substituted with any other directory. The following subdirectories will be created:
- tensorboard_random_presample: contains logs from random sampling to fill up the initial replay buffer
- tensorboard: contains logs from the training
- agent_checkpoints: contains agent state dict files
- bestiary_renders: contains svg renders of the highest reward episodes (best viewed with white BG setting in the image viewer)
- bestiary_renders_random: same but for the presampling run

###Evaluating an Agent
To run evaluation of the agent, the following command should be used:
``` 
python eval_PGFS.py "./data" "./data/agent_checkpoints/agent9000.state" "./gym_PGFS/configs/config_server_default.yaml" 100 "./fig.png"
```
where the first argument can be substituted with the path to the agent state file and the second one with the number of episodes to test over.


### Current Work

#### Stage 1:
- ✅ adapt to using guacamole utility functions for consistency:
  - ❎ use guacamol functions for preprocessing
  - 🟩 improve molecule filtering using guacamole routines to improve stability
- 🟥 improve performance and paralellize:
  - 🟥 implement the distributed architecture for training an agent
  - 🟨 implement distributed architecture for hyperparameter selection
  - 🟥 get rid of the dusk dependency?
  - 🟥 improve configurability, make compatible with parallel computation
  - 🟨 consider replacing tensorboard with aimstack
- 🟨 implement methods from Renz et al. paper: 
  - 🟩 rework the scoring function mechanism to be more flexible
  - 🟩 implement a scoring function for the PGFS similar to one in mgen_fail
  - 🟨 implement an evaluation routine for the PGFS

#### Stage 2:
- 🟥 integrate one of the exploration techniques into the algorithm.


🟥🟧🟨🟩🟦🟪🟫☑✅❎