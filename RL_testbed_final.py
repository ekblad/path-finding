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

def shortest_path(weighting_scheme, samples, Hueristic_df, attributes,dir_path):

	#initialize variables
	num_samples = len(samples)
	accuracy = np.zeros(num_samples)
	for z,sample in enumerate(samples):
		#get data for each sample 
		with h5py.File(dir_path + '/Sample_A/' + sample + '/' + 'EdgeCoordinates.hdf5','r') as f:
			EdgeCoordinates = f.get('default')[()]

		#determine edge weights from weighting scheme
		sample_hueristics = Hueristic_df.loc[Hueristic_df['SampleName'] == sample, attributes].iloc[0::2].values
		edge_weights = np.sum(weighting_scheme.T*sample_hueristics, axis = 1)
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
		PT = sio.loadmat(dir_path + '/All_Samples_001/Eroding_'+ sample[-3:-1] +'/'+ sample + '.mat')
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
		accuracy[z] = ((np.sum(Overlapping_Matrix))/(np.sum(Shortest_Path)))

	return np.mean(accuracy) 

def q_linear(weights, state, action_indx, action_stp, x_sample): 
	state_fn = np.copy(state)
	state_fn[action_indx] = state_fn[action_indx] + action_stp
	ret_ = np.mean(np.array([np.inner(weights,(state_fn*np.mean(samp, axis = 0))) for samp in x_sample]))
	return ret_

def q_grad_linear(state, action_indx, action_stp, x_sample):
	state_fn = np.copy(state)
	state_fn[action_indx] = state_fn[action_indx] + action_stp
	ret_grad = np.mean(np.array([np.inner(state_fn,np.mean(samp,axis=0)) for samp in x_sample]))
	return ret_grad 

def next_action(state_val, action_step):
	state_next = state_val + action_step
	if state_next < 0:
		return action_step - state_next
	elif state_next > 10:
		return action_step - (state_next-10)
	else:
		return action_step

def semigradient_sarsa_batch(episodes,Samples,attributes,Hueristic_df,alpha,gamma,epsilon,batchsize,action_num,action_disc,dir_path,max_steps,trial):    
	np.random.seed(seed=trial)
	rewards_store = {}
	states_store = {}
	actions_store = {}
	weights_store = {}
	weight_track = []
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

		action_step = np.random.choice(action_mags)*action_disc        
		action = next_action(state[action_index],action_step)

		# action_vect[action_index] = action_step*nextaction
		action_vect[action_index] = action
		actions.append(action_vect.copy())

		i = 0  #reset step count
		epsilon_func = epsilon
		alpha_func = alpha
		
		while term == False:
			term = True 
			i += 1

			#collect feature data for random samples            
			sample_batch = np.random.choice(Samples,batchsize, replace = False) # draw batchsize random samples          
			x_samp = [Hueristic_df.loc[Hueristic_df['SampleName']==sample, attributes].values[0::2,] for sample in sample_batch]
			
			#evaluate S and S' on same samples
			state_prime = state.copy()
			state_prime[action_index] = state_prime[action_index] + action

			#determine and store reward for the given state and samples
			reward = shortest_path(state_prime, sample_batch, Hueristic_df, attributes, dir_path) #change this to -(1-Accuracy) (i.e. minimize loss)
			rewards.append(reward)

			if any(np.abs(action_vect[~np.isnan(action_vect)]) >= 10e-15) or any(np.isnan(action_vect)): # checking action for termination criteria is same as checking state it takes you to
				term = False

				# epsilon-greedy action selection
				if np.random.random_sample() < epsilon_func:
					action_prime_index = np.random.randint(num_attributes)
					action_prime_step = np.random.choice(action_mags)*action_disc    
					action_prime = next_action(state_prime[action_prime_index], action_prime_step)
				else:
					q_enum = np.zeros((num_attributes,len(action_mags)))
					action_prime_store = np.zeros((num_attributes,len(action_mags)))
					for j in range(num_attributes):
						for k,mag in enumerate(action_mags*action_disc):
							action_prime = next_action(state_prime[j], mag)
							action_prime_store[j,k] = action_prime
							q_enum[j,k] = q_linear(weights,state_prime,j,action_prime,x_samp)

					#find max values
					max_values = [(q_enum[j,k], action_prime_store[j,k], j) for k in range(len(action_mags)) for j in range(num_attributes) if q_enum[j,k] == np.nanmax(q_enum)]
					#if more than one max value choose randomly
					if len(max_values) > 1: 
						q_max, action_prime, action_prime_index = max_values[np.random.randint(len(max_values))]
					else:
						q_max, action_prime, action_prime_index = max_values[0]        
					#if no change to the state pick the attribute with the highest sum q_hat over all actions
					if action_prime == 0:
						action_prime_index = q_enum.sum(axis=1).argmax()

				q = q_linear(weights, state, action_index, action, x_samp)
				q_prime = q_linear(weights, state_prime, action_prime_index, action_prime, x_samp)

				q_grad = q_grad_linear(state, action_index, action, x_samp)
				weights[action_index] = np.add(weights[action_index],alpha_func*np.multiply(reward + gamma * np.subtract(q_prime,q),q_grad))
				
				state = state_prime.copy()
				states.append(state.copy())
	
				action = action_prime
				action_index = action_prime_index
				action_vect[action_index] = action
				actions.append(action_vect.copy())
	
			else:
				q = q_linear(weights, state, action_index, action, x_samp)
				q_grad = q_grad_linear(state, action_index, action, x_samp)
				weights[action_index] = weights[action_index] + np.multiply(alpha_func*(reward - q),q_grad)
				term = True
	
			if i > max_steps: #CHANGE BACK TO 100
				term = True
			weight_track.append(weights.copy())
			
		rewards_store[episode] = rewards.copy()
		states_store[episode] = states.copy()
		actions_store[episode] = actions.copy()
		weights_store[episode] = weights.copy()
	
	params = dict(episodes=episodes,samples=Samples,attributes=attributes,alpha=alpha,gamma=gamma,
					epsilon=epsilon,batchsize=batchsize,action_num=action_num,action_disc=action_disc,
					max_steps=max_steps,trial=trial)
	
	results = dict(rewards=rewards_store,states=states_store,actions=actions_store,weights=weights_store,weight_track = weight_track, params=params)
	with open('sarsa_batch_results_{}.pickle'.format(trial), 'wb') as f:
		# Pickle the results dictionary using the highest protocol available.
		pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

	return weight_track,rewards_store,states_store,actions_store,weights_store # all indexed/keyed by episode int

def batch():
	tic = time.time()
	dir_path = os.path.dirname(os.path.realpath(__file__)) # where to write results pickle

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
	episodes = 15 # number of episodes
	attributes = ['ArcLength', 'MeanWidth', 'LongandThick', 'Curvature', 'Connectivity']
	alpha = 0.2 # initial step size for each episode
	gamma = 1 # undiscounted
	epsilon = 0.2
	batchsize = 4 # number of samples at each step
	action_num = 4 # dimension of action space (not including 0)
	action_disc = 0.5 # centered at 0, steps of this to either side
	max_steps = 1000
	trial = 12
	
	local_path = os.path.join(dir_path,'sarsa_batch_results_{}'.format(trial))
	if 'sarsa_batch_results_{}'.format(trial) in os.listdir(dir_path):
		rmtree('sarsa_batch_results_{}'.format(trial)) # only turn on if need to do again
	# os.mkdir(local_path, 755)
	os.mkdir(local_path)
	os.chdir(local_path)
	weights_tracker,rewards_store,states_store,actions_store,weights_store = semigradient_sarsa_batch(episodes,Samples,attributes,Hueristic_tr,alpha,gamma,epsilon,batchsize,action_num,action_disc,dir_path,max_steps,trial)
	fig, ax = plt.subplots(4,1,figsize=(5,10)) 
	state_tracker = []
	for i in range(episodes):
		state_tracker +=states_store[i] 
	attr_plot  = [states_store[i][-1] for i in states_store]
	a_store = pd.DataFrame(attr_plot,columns=attributes)
	s_track_store = pd.DataFrame(state_tracker,columns=attributes)
	w_track_store = pd.DataFrame(weights_tracker,columns=attributes)
	w_store = pd.DataFrame.from_dict(weights_store,orient='index')
	w_store.columns = attributes

	for i in w_store.columns:
		ax[0].scatter(np.arange(0,len(w_store)),w_store[i],label=i)
		ax[1].scatter(np.arange(0,len(a_store)),a_store[i],label=i)
		ax[2].scatter(np.arange(0,len(w_track_store)),w_track_store[i],label=i)
		ax[3].scatter(np.arange(0,len(s_track_store)),s_track_store[i],label=i)

	ax[1].set_xlabel('Episodes')
	ax[3].set_xlabel('Steps')
	ax[0].set_ylabel('q_hat Weights')
	ax[1].set_ylabel('Attribute Weights')
	ax[2].set_ylabel('q_hat Weights')
	ax[3].set_ylabel('Attribute Weights')
	
	handles, labels = ax[1].get_legend_handles_labels()
	fig.legend(handles=handles,labels=labels,frameon=False,bbox_to_anchor=(1.35,0.5),loc='right')
	plt.savefig('weighting_scheme.png',format='png',bbox_inches='tight',dpi=300)
	os.chdir(dir_path)
	toc = time.time()
	print('runtime = ', toc - tic)

def semigradient_sarsa_continuous(Samples,attributes,Hueristic_df,alpha,beta,gamma,epsilon,batchsize,action_num,action_disc,dir_path,max_steps=100,trial=0):    
	np.random.seed(seed=trial)
	rewards_store = {}
	avg_rewards = {}
	avg_rewards[0] = 0
	states_store = {}
	actions_store = {}
	weights_store = {}
	action_mags = np.arange(int(-action_num/2),int(action_num/2)+1) # e.g. [-1.  0.  1.]  
	num_attributes = len(attributes)

	#initialize random weights 
	weights = np.add(np.zeros(num_attributes),np.random.standard_normal(num_attributes))

	term = False
	i = 0  #reset step count
	state = np.random.uniform(0,10,num_attributes) #random initial state
	states_store[i] = state.copy()
	
	action_vect = np.full((num_attributes,), np.nan)    
	action_index = np.random.randint(num_attributes)

	action_step = np.random.choice(action_mags)*action_disc		
	action = next_action(state[action_index],action_step)

	action_vect[action_index] = action
	actions_store[i] = action_vect.copy()

	epsilon_func = epsilon
	alpha_func = alpha
	
	while term == False:
		term = True 
		print(i,action_vect)

		#collect feature data for random samples            
		sample_batch = np.random.choice(Samples,batchsize, replace = False) # draw batchsize random samples          
		x_samp = [Hueristic_df.loc[Hueristic_df['SampleName']==sample, attributes].values[0::2,] for sample in sample_batch]
		
		#evaluate S and S' on same samples
		state_prime = state.copy()
		state_prime[action_index] = state_prime[action_index] + action

		#determine and store reward for the given state and samples 
		reward = shortest_path(state_prime, sample_batch, Hueristic_df, attributes, dir_path)
		rewards_store[i] = reward
		i += 1
		if any(np.abs(action_vect[~np.isnan(action_vect)]) >= 10e-15) or any(np.isnan(action_vect)): # checking action for termination criteria is same as checking state it takes you to
			term = False

			# epsilon-greedy action selection
			if np.random.random_sample() < epsilon_func:
				action_prime_index = np.random.randint(num_attributes)
				action_prime_step = np.random.choice(action_mags)*action_disc	
				action_prime = next_action(state_prime[action_prime_index], action_prime_step)
			else:
				q_enum = np.zeros((num_attributes,len(action_mags)))
				action_prime_store = np.zeros((num_attributes,len(action_mags)))
				for j in range(num_attributes):
					for k,mag in enumerate(action_mags*action_disc):
						action_prime = next_action(state_prime[j], mag)
						action_prime_store[j,k] = action_prime
						q_enum[j,k] = q_linear(weights,state_prime,j,action_prime,x_samp)

				#find max values
				max_values = [(q_enum[j,k], action_prime_store[j,k], j) for k in range(len(action_mags)) for j in range(num_attributes) if q_enum[j,k] == np.nanmax(q_enum)]
				#if more than one max value choose randomly
				if len(max_values) > 1: 
					q_max, action_prime, action_prime_index = max_values[np.random.randint(len(max_values))]
				else:
					q_max, action_prime, action_prime_index = max_values[0]        
				#if no change to the state pick the attribute with the highest sum q_hat over all actions
				if action_prime == 0:
					action_prime_index = q_enum.sum(axis=1).argmax()

			q = q_linear(weights, state, action_index, action, x_samp)
			q_prime = q_linear(weights, state_prime, action_prime_index, action_prime, x_samp)

			delta = reward - avg_rewards[i-1] + q_prime - q
			avg_rewards[i] = avg_rewards[i-1] + beta*delta

			q_grad = q_grad_linear(state, action_index, action, x_samp)
			weights[action_index] = weights[action_index] +  alpha_func*delta*q_grad
			state = state_prime.copy()
			states_store[i] = state.copy()

			action = next_action(state[action_prime_index],action_prime)
			action_index = action_prime_index
			action_vect[action_index] = action
			actions_store[i] = action_vect.copy()

		if i > max_steps:
			term = True
		weights_store[i] = weights.copy()
	
	params = dict(samples=Samples,attributes=attributes,alpha=alpha,gamma=gamma,beta=beta,
					epsilon=epsilon,batchsize=batchsize,action_num=action_num,action_disc=action_disc,
					max_steps=max_steps,trial=trial)
	
	results = dict(rewards=rewards_store,avg_rewards=avg_rewards,states=states_store,actions=actions_store,weights=weights_store,params=params)
	with open('sarsa_cont_results_{}.pickle'.format(trial), 'wb') as f:
		# Pickle the results dictionary using the highest protocol available.
		pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

	return weights,rewards_store,avg_rewards,states_store,actions_store,weights_store # all indexed/keyed by episode int

def continuous():
	tic = time.time()
	dir_path = os.path.dirname(os.path.realpath(__file__)) # where to write results pickle

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
	attributes = ['ArcLength', 'MeanWidth', 'LongandThick', 'Curvature', 'Connectivity']
	alpha = 0.2 # initial step size for each episode
	beta = 0.1 # average update size
	gamma = 1 # undiscounted
	epsilon = 0.2
	batchsize = 1 # number of samples at each step
	action_num = 4 # dimension of action space (not including 0)
	action_disc = 0.2 # centered at 0, steps of this to either side
	max_steps = 1000
	trial = 12
	
	local_path = os.path.join(dir_path,'sarsa_cont_results_{}'.format(trial))
	if 'sarsa_cont_results_{}'.format(trial) in os.listdir(dir_path):
		rmtree('sarsa_cont_results_{}'.format(trial)) # only turn on if need to do again
	# os.mkdir(local_path, 755)
	os.mkdir(local_path)
	os.chdir(local_path)
	weights,rewards_store,avg_rewards,states_store,actions_store,weights_store = semigradient_sarsa_continuous(Samples,attributes,Hueristic_tr,alpha,beta,gamma,epsilon,batchsize,action_num,action_disc,dir_path,max_steps,trial)
	fig, ax = plt.subplots(2,1,figsize=(5,7)) 

	a_store = pd.DataFrame.from_dict(states_store,orient='index')
	a_store.columns = attributes	
	w_store = pd.DataFrame.from_dict(weights_store,orient='index')
	w_store.columns = attributes

	for i in w_store.columns:
		ax[0].plot(np.arange(0,len(w_store)),w_store[i],label=i)
		ax[1].plot(np.arange(0,len(a_store)),a_store[i],label=i)

	ax[1].set_xlabel('Steps')
	ax[0].set_ylabel('q_hat Weights')
	ax[1].set_ylabel('Attribute Weights')
	handles, labels = ax[1].get_legend_handles_labels()
	fig.legend(handles=handles,labels=labels,frameon=False,bbox_to_anchor=(1.35,0.5),loc='right')
	plt.savefig('weighting_scheme.png',format='png',bbox_inches='tight',dpi=300)

	fig = plt.figure()
	ax = fig.add_subplot(111)

	r_store = pd.DataFrame.from_dict(rewards_store,orient='index')
	r_store.columns = ['Rewards']	
	ra_store = pd.DataFrame.from_dict(avg_rewards,orient='index')
	ra_store.columns = ['Average Rewards']

	r_store.plot(y='Rewards',kind='line',ax=ax)
	ra_store.plot(y='Average Rewards',kind='line',ax=ax)

	ax.set_xlabel('Steps')
	ax.set_ylabel('Reward')

	plt.savefig('rewards.png',format='png',bbox_inches='tight',dpi=300)

	os.chdir(dir_path)
	toc = time.time()
	print('runtime = ', toc - tic)

if __name__ == '__main__':
	# batch()
	continuous()

