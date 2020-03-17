#DONT FORGET TO DOWNLOAD HUERISTIC CSV INTO MAIN FOLDER
import numpy as np
import pandas as pd
import h5py
import pickle
import os
import networkx as nx
import scipy.io as sio
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from shutil import rmtree
np.random.seed(seed=15)

def shortest_path(weighting_scheme, samples, Hueristic_df, attributes):
#    print(weighting_scheme)
    #initialize variables
    num_samples = len(samples)
    accuracy = np.zeros(num_samples)
    for z,sample in enumerate(samples):
        #get data for each sample 
        with h5py.File('Sample_A/' + sample + '/' + 'EdgeCoordinates.hdf5','r') as f:
            EdgeCoordinates = f.get('default')[()]

        #determine edge weights from weighting scheme
        sample_hueristics = Hueristic_df.loc[Hueristic_df['SampleName'] == sample, attributes].iloc[0::2].values
        edge_weights = np.sum(weighting_scheme.T*sample_hueristics, axis = 1)
#        print(min(edge_weights))
        num_edges = len(edge_weights)

        sample_nodes = Hueristic_df.loc[Hueristic_df['SampleName'] == sample,'nodeID'].values.astype(np.int)

        #node ID for source node
        source = np.min(sample_nodes)
        #node ID for target node                      
        target = np.max(sample_nodes)

        #indexing sample nodes 
        nodelist1 = sample_nodes[0::2]
        nodelist2 = sample_nodes[1::2]

        #build network from sparselist
        UG = nx.Graph()
        for i in range(num_edges):
            UG.add_edge(nodelist1[i], nodelist2[i], weight = edge_weights[i])
        try:
            path = nx.dijkstra_path(UG, source, target)
        except nx.NetworkXNoPath:
            print(sample, 'has no path!!!')
            continue

        voxel = 25e-6 #[m]
        #get x & y coordinates of the path
        X = []
        Y = []
        for i in range(1,len(path[1:-2])):
            node1candidates = np.where(sample_nodes == path[i])
            node2candidates = np.where(sample_nodes == path[i+1])
            for node1 in node1candidates[0]:
                for node2 in node2candidates[0]:
                    edge_node1 = Hueristic_df.iloc[node1,6]
                    edge_node2 = Hueristic_df.iloc[node2,6]
                    if edge_node1 == edge_node2:
                        X = np.append(X,EdgeCoordinates[np.where(EdgeCoordinates[:,0]==edge_node1),1]/voxel)
                        Y = np.append(Y,EdgeCoordinates[np.where(EdgeCoordinates[:,0]==edge_node2),2]/voxel)

        #import true percolating path for this sample
        PT = sio.loadmat('All_Samples_001/Eroding_'+ sample[-3:-1] +'/'+ sample + '.mat')
        PT = PT['plotting_final']
        lower = 0
        upper = 1
        threshold = 3
        Percolating_Path = np.where(PT>=threshold, upper, lower)

        # create zeros matrix to later add the nodes in the shortest path to
        Shortest_Path = np.zeros(PT.shape)  
        #renaming X so that there are no issues when the code loops 
        y1=X.astype(int)
        #renaming Y so that there are no issues when the code loops   
        x1=Y.astype(int)

        #adding the nodes to the zero matrix
        for i in range(1, len(x1)-1):      
            Shortest_Path[x1[i],y1[i]]=1

        #calculate accuraccy comparing predicted path and percolating path
        Overlapping_Matrix = np.multiply(Shortest_Path, Percolating_Path)
        accuracy[z] = 100*((np.sum(Overlapping_Matrix))/(np.sum(Shortest_Path)))

    return np.mean(accuracy) 

def q_linear(weights, state, action_indx, action_stp, x_sample): 
    state_fn = np.copy(state)
    state_fn[action_indx] = state_fn[action_indx] + action_stp
    return np.mean(np.array([np.inner(weights,(state_fn*np.mean(samp, axis = 0))) for samp in x_sample]))

def q_grad_linear(state, action_indx, action_stp, x_sample):
    state_fn = np.copy(state)
    state_fn[action_indx] = state_fn[action_indx] + action_stp
    return np.mean(np.array([np.inner(state_fn,np.mean(samp,axis=0)) for samp in x_sample]),axis=0) 

def q_neural(): # placeholder
    return 1

def next_actions(state, action_index, action_mags, action_disc):
    actions = action_mags*action_disc
    if state[action_index] == 10: 
        actions[-1] = np.nan
    elif state[action_index] == 0: 
        actions[0] = np.nan
    elif state[action_index] + actions[-1] > 10:
        actions[-1] = 10 - state[action_index]
        print('over 10', actions,  state[action_index], action_index)
    elif state[action_index] + actions[0] < 0: 
        actions[0] = -state[action_index]
    return actions    

def semigradient_sarsa_batch(episodes,Samples,attributes,Hueristic_df,alpha,gamma,epsilon,batchsize,action_num,action_disc,trial=0):    
    np.random.seed(seed=trial)
    rewards_store = {}
    states_store = {}
    actions_store = {}
    weights_store = {}
    action_mags = np.arange(int(-action_num/2),int(action_num/2)+1) # e.g. [-1.  0.  1.]  
    num_attributes = len(attributes)
    #initialize random weights 
    weights = np.add(np.zeros(num_attributes),np.random.standard_normal(num_attributes))
    for episode in range(episodes):
        term = False
        rewards = []
        states = []
        actions = []
        state = np.random.uniform(0,10,num_attributes) #random initial state
        states.append(state.copy())
        
        action_vect = np.full((num_attributes,), np.nan)    
        action_index = np.random.randint(num_attributes)
        nextactions = next_actions(state, action_index, action_mags, action_disc)
        action_step = np.random.choice(nextactions[~np.isnan(nextactions)]).astype(np.int) # random init. action
        action_vect[action_index] = action_step
        actions.append(action_vect.copy())
        if episode > 0: 
            action_disc = action_disc/episode # anneal action discretization
        i = 0  #reset step count
        epsilon_func = epsilon
        alpha_func = alpha
        state_prime = state 
        while term == False:
    #         term = True 
            i += 1
            # anneal action discretization
            action_disc -= 1/100
            epsilon_func -= 1/100 #epsilon is the amount of random exploration
            alpha_func -= 1/100
            #collect feature data for random samples            
            sample_batch = np.random.choice(Samples,batchsize, replace = False) # draw batchsize random samples          
            x_samp = [Hueristic_df.loc[Hueristic_df['SampleName']==sample, attributes].values[0::2,] for sample in sample_batch]
            
            #evaluate S and S' on same samples
            state_prime[action_index] = state_prime[action_index] + action_step

            #determine and store reward for the given state and samples 
            reward = shortest_path(state, sample_batch, Hueristic_df, attributes) #change this to -(1-Accuracy) (i.e. minimize loss)
            rewards.append(reward)
            if any(action_vect != 0): # checking action for termination criteria is same as checking state it takes you to
                term = False
                # epsilon-greedy action selection
                if np.random.random_sample() < epsilon_func:
#                    print('random')
                    action_prime_index = np.random.randint(num_attributes)
                    nextactions = next_actions(state, action_prime_index, action_mags, action_disc)
                    action_prime_step = np.random.choice(nextactions[~np.isnan(nextactions)]).astype(np.int) # random init. action
                    action_vect[action_prime_index] = action_prime_step
                else:
                    q_enum = np.zeros((num_attributes,len(action_mags)))
                    action_step_enum = np.zeros((num_attributes,len(action_mags)))
                    for j in range(num_attributes):
                        nextactions = next_actions(state_prime, j, action_mags, action_disc)
                        for k, direction in enumerate(nextactions):
                            action_step_enum[j,k] = direction
                            if direction == np.nan:
                                q_enum[j,k] = np.nan
                            else:
                                q_enum[j,k] = np.sum(q_linear(weights,state_prime,j,direction,x_samp))
                    #find max values
                    max_values = [(q_enum[j,k], action_step_enum[j,k], j) for k in range(len(action_mags)) for j in range(num_attributes) if q_enum[j,k] == np.nanmax(q_enum)]
                    #if more than one max value choose randomly
                    if len(max_values) > 1: 
                        q_max, action_prime_step, action_prime_index = max_values[np.random.randint(len(max_values))]
                    else:
                        q_max, action_prime_step, action_prime_index = max_values[0]        
                    #if no change to the state pick the attribute with the highest sum q_hat over all actions
                    if action_prime_step == 0:
                        action_prime_index = q_enum.sum(axis=1).argmax()  
                q = q_linear(weights, state, action_index, action_step, x_samp)
                q_prime = q_linear(weights, state_prime, action_prime_index, action_prime_step, x_samp)

                q_grad = q_grad_linear(state, action_index, action_step, x_samp)
                weights = np.add(weights,alpha_func*np.multiply(reward + gamma * np.subtract(q_prime,q),q_grad))
                
                state = state_prime
                states.append(state.copy())
    
                action_step = action_prime_step
                action_index = action_prime_index
                action_vect[action_index] = action_step
                actions.append(action_vect.copy())
    
            else:
                q = q_linear(weights, state, action_index, action_step, x_samp)
                q_grad = q_grad_linear(state, action_index, action_step, x_samp)
                weights = weights + np.multiply(alpha_func*(reward - q),q_grad)
    
            if i > 100: #CHANGE BACK TO 100
                term = True

        rewards_store[episode] = rewards
        states_store[episode] = states
        actions_store[episode] = actions
        weights_store[episode] = weights
    
    params = dict(episodes=episodes,samples=Samples,attributes=attributes,alpha=alpha,gamma=gamma,
            epsilon=epsilon,batchsize=batchsize,action_num=action_num,action_disc=action_disc)
    
    results = dict(rewards=rewards,states=states,actions=actions,weights=weights,params=params)
    print(results)
    with open('sarsa_batch_results_{}.pickle'.format(trial), 'wb') as f:
        # Pickle the results dictionary using the highest protocol available.
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    return weights,rewards_store,states_store,actions_store,weights_store # all indexed/keyed by episode int

def semigradient_sarsa_continuous(episodes,Samples,attributes,alpha,gamma,epsilon,batchsize,action_num,action_disc,trial=0):  
    return 1

def main():

    tic = time.time()
#    dir_path = os.path.dirname(os.path.realpath(__file__)) # where to write results pickle

    #read in hueristic sample data
    Hueristic_data = pd.read_csv('heuristic_info_all_samples.csv')
    samples = Hueristic_data['SampleName'].unique()
    num_samples = len(samples)
    
    #split samples into training and test data set 
    training_index = np.random.choice(num_samples, int(num_samples*.8), replace = False)
    Hueristic_tr = Hueristic_data.loc[Hueristic_data['SampleName'].isin(samples[training_index]),:].copy()
    Hueristic_te = Hueristic_data.loc[~Hueristic_data['SampleName'].isin(samples[training_index]),:].copy()
    
    scaler = MinMaxScaler()
    Hueristic_tr.iloc[:,7:] = scaler.fit_transform(Hueristic_tr.iloc[:,7:])
    Samples = Hueristic_tr['SampleName'].unique()

    #algorithm parameters
    episodes = 2 # number of episodes
    attributes = ['ArcLength', 'MeanWidth', 'LongandThick', 'Curvature', 'Connectivity']
    alpha = 1. # initial step size for each episode
    gamma = 1 # undiscounted
    epsilon = 0.5
    batchsize = 3 # number of samples at each step
    action_num = 2 # dimension of action space (not including 0)
    action_disc = 1.0 # centered at 0, steps of this to either side
    trial = 1
    
    
#    local_path = os.path.join(dir_path,'sarsa_batch_results_{}'.format(trial))
#    if 'sarsa_batch_results_{}'.format(trial) in os.listdir(dir_path):
#            rmtree('sarsa_batch_results_{}'.format(trial)) # only turn on if need to do again
#    os.mkdir(local_path)
#    os.chdir(local_path)
    weights,rewards_store,states_store,actions_store,weights_store = semigradient_sarsa_batch(episodes,Samples,attributes,Hueristic_tr,alpha,gamma,epsilon,batchsize,action_num,action_disc,trial)
    fig, ax = plt.subplots(2,1,figsize=(5,7)) 
    # ax.set_aspect('equal')

    attr_plot  = [states_store[i][-1] for i in states_store]
    a_store = pd.DataFrame(attr_plot,columns=attributes)
    w_store = pd.DataFrame.from_dict(weights_store,orient='index')
    w_store.columns = attributes
    # print(w_store.shape)
    for i in w_store.columns:
        # print(w_store[:,:,i])
        ax[0].scatter(np.arange(0,len(w_store)),w_store[i],label=i)
        ax[1].scatter(np.arange(0,len(a_store)),a_store[i],label=i)
    # ax.set_ylim(0,30)
    ax[1].set_xlabel('Episodes')
    ax[0].set_ylabel('q_hat Weights')
    ax[1].set_ylabel('Attribute Weights')
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles=handles,labels=labels,frameon=False,bbox_to_anchor=(1.35,0.5),loc='right')
    plt.savefig('weighting_scheme.png',format='png',bbox_inches='tight',dpi=300)
#    os.chdir(dir_path)
    toc = time.time()
    print('runtime = ', toc - tic)

if __name__ == '__main__':
    main()

