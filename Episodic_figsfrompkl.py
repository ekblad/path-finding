import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

results = pd.read_pickle('sarsa_batch_results_12/sarsa_batch_results_12.pickle')

#seperate out key data inputs
keys = list(results.keys())
attributes = results['params']['attributes']
num_episodes = len(results['rewards'])

rewards = np.array([])
for i in range(num_episodes):
    rewards = np.append(rewards, np.array(results['rewards'][i]))

#combine data into single record by step
states = []
for i in range(num_episodes):
    states+= results['states'][i]

actions = []
for i in range(num_episodes):
    actions+= results['actions'][i]

weights = []
for i in range(num_episodes):
    weights.append(results['weights'][i])

ep_len = []
for i in range(num_episodes):
    ep_len.append(len(results['states'][i]))

#determine number of steps
num_steps = np.array(states).shape[0]

#get tracked weight data
weight_track = results['weight_track'][:num_steps ]

mean_reward = rewards[ep_index[-1]:].mean()

#make data into dataframes
state_df = pd.DataFrame(data = states, columns = attributes)
weights_df = pd.DataFrame(data = weight_track, columns = attributes)
num_cols = state_df.shape[1]

#plot Q_hat and Sample results by steps
fig, ax = plt.subplots(num_cols,2,figsize=(10,8))
cmap = plt.get_cmap('tab10')
for i in range(num_cols):
    weights_df.plot(kind='line', y = attributes[i], ax = ax[i,0], color = cmap(i))
    ax[i,0].set_ylabel('q_hat Weight')
    state_df.plot(kind='line', y = attributes[i], ax = ax[i,1], color = cmap(i))
    ax[i,1].set_ylabel('Attribute Weight')

ax[num_cols-1,0].set_xlabel('Steps')
ax[num_cols-1,1].set_xlabel('Steps')
plt.suptitle('Episodic Sarsa: Batch Size = 4')
fig.tight_layout()
fig.subplots_adjust(top=.94)
plt.savefig('weighting_scheme_batch4.png',format='png',bbox_inches='tight',dpi=300)
