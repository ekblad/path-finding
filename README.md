# path-finding
This is a repository for a class project in reinforcement learning.

## Data can be found here:
[here](https://drive.google.com/drive/folders/13a9KV83qzcuvzcp1D21rtZk1au6wvNNy?usp=sharing "UC Davis data access")

### Run the file RL_testbed_final.py from the same directory as the following: 
1. Sample_A (data folder) 
2. heuristic_info_all_samples.csv

Creates a folder for results (pickled dictionary) and figs based on a trial number. The trial number is also taken as the random seed.

### Two algorithms implemented:
1. Episodic semigradient SARSA (Sutton and Barto, pg. 244) - with linear approximation function 
2. Continuous semigradient SARSA (Sutton and Barto, pg. 251) - with linear approximation function
