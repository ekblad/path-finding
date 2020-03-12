
import numpy as np
import pandas as pd
import h5py
import pickle
from matplotlib.colors import ListedColormap
import os
import networkx as nx
import scipy.io as sio
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import normalize
from shutil import rmtree
np.random.seed(seed=15)

def heuristics(sample,dir_path):
	"""Calculate the following 14 Normalized Hueristic Measurements for sample. 
	[mean width, inverted mean width, arclength, euclidean distance, minimum width, inverted minimum width, long and thick, 
	widest bottelneck, quino, curverature, straightness, volume, aspect ratio, poisueille and connectivity]"""
	#read in file 
	with h5py.File(dir_path+'/Sample_A/'+sample+'/'+'sparselist.hdf5','r') as f:
		# print(list(f.keys()))
		sparselist = f.get('default')[()]
		
	#Determine 14 channel attributes from sample (Generally these equations should be checked against definitions)
	meanWidth = sparselist[0::2,7]                                      #H1
	inv_meanWidth = np.reciprocal(meanWidth)                            #H2
	ArcLength = sparselist[0::2,5]                                      #H3
	EuclideanDistance = sparselist[0::2,4]                              #H4
	MinWidth = sparselist[0::2,8]                                       #H5
	LongandThick = ArcLength/meanWidth                                  #H6
	widestBottleneck = np.reciprocal(MinWidth)                          #H7
	Quino = np.exp(-np.reciprocal(meanWidth))                           #H8
	Curvature = (EuclideanDistance**2)/ArcLength                        #H9  -> changed based on powerpoint
	Straightness = np.reciprocal(Curvature, where = (Curvature != 0))   #H10
	Volume = ArcLength*((MinWidth)**2)*np.pi                            #H11
	AspectRatio = ArcLength/MinWidth                                    #H12
	Poiseuille = (meanWidth**4)/ArcLength                               #H13
	Connectivity = sparselist[0::2,6]                                   #H14       
	
	#save into one large matrix
	attributes = np.vstack((ArcLength, EuclideanDistance, MinWidth,meanWidth, inv_meanWidth, LongandThick, widestBottleneck,
							Quino,Curvature, Straightness, Volume, AspectRatio, Poiseuille, Connectivity))

	# attributes =  np.divide((attributes.T - np.mean(attributes,axis=1)),np.std(attributes,axis=1)).T # standardize data
	attributes =  np.divide((attributes.T ),np.std(attributes,axis=1)).T
	# print(attributes.shape)    
	#return normalized matrix 
	# return normalize(attributes, norm = 'max')
	return attributes

def shortest_path(weighting_scheme,Samples,dir_path,plot=False, all_samples = True):#w,Samples,plot=False):
	if any(weighting_scheme>10):
		return 1
	else:
		num_samples = len(Samples)
		accuracy = np.zeros(num_samples)
		for z in range(num_samples):
			#get data for each sample 
			with h5py.File(dir_path+'/Sample_A/'+Samples[z]+ '/' + 'sparselist.hdf5','r') as f:
				sparselist = f.get('default')[()]
			with h5py.File(dir_path+'/Sample_A/'+Samples[z]+'/'+'EdgeCoordinates.hdf5','r') as f:
				EdgeCoordinates = f.get('default')[()]
			
			#determine edge weights from weighting scheme
			attributes = heuristics(Samples[z],dir_path)

			edge_weights = np.transpose(np.matmul(np.abs(weighting_scheme),attributes))
			num_edges = len(edge_weights)
			
			#node ID for source node
			srce = np.min(sparselist[:,0])
			#node ID for target node                      
			targ = np.max(sparselist[:,0]).astype(dtype = np.int)
			#indexing of sparse list
			l_1 = sparselist[0::2,0] ###FYI this sparse list sometimes returns negatives (not sure if this is an issue...)
			l_2 = sparselist[1::2,0] ###FYI this sparse list sometimes returns negatives (not sure if this is an issue...)
			
			#build network from sparselist
			UG = nx.Graph()
			for i in range(num_edges):
				UG.add_edge(int(l_1[i]), int(l_2[i]), weight = edge_weights[i])
			try:
				path = nx.dijkstra_path(UG, srce, targ)
			except nx.NetworkXNoPath:
				print(Samples[z], 'has no path!!!')
				continue
				
			voxel = 25e-6 #[m]
			#get x & y coordinates of the path
			X = []
			Y = []
			for i in range(1,len(path[1:-2])):
				node1candidates = np.where(sparselist[:,0] == path[i])
				node2candidates = np.where(sparselist[:,0] == path[i+1])
	#             print(node1candidates[0], node2candidates[0], type(node1candidates[0]))
				for node1 in node1candidates[0]:
					for node2 in node2candidates[0]:
						if sparselist[node1,11] == sparselist[node2,11]:
							X = np.append(X,EdgeCoordinates[np.where(EdgeCoordinates[:,0]==sparselist[node1,11]),1]/voxel)
							Y = np.append(Y,EdgeCoordinates[np.where(EdgeCoordinates[:,0]==sparselist[node2,11]),2]/voxel)
			
			#import true percolating path for this sample
			PT = sio.loadmat(dir_path+'/All_Samples_001/Eroding_'+ Samples[z][-3:-1] +'/'+ Samples[z] + '.mat')
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
				
		return 1 - np.mean(accuracy)

def semigradient_sarsa_batch(episodes,Samples,attributes,alpha,gamma,epsilon,batchsize,action_num,action_disc,dir_path,trial=0):    
	np.random.seed(seed=trial)	
	rewards_store = {}
	states_store = {}
	actions_store = {}
	weights_store = {}
	
	weights = np.add(np.zeros(attributes)+5,np.random.standard_normal(attributes))	
	for episode in range(episodes):
		term = False
		i = 0

		alpha_ = alpha
		rewards = []

		states = []
		state = np.random.uniform(0,10,attributes)
		states.append(state)

		actions = []
		action_mags = np.arange(0-int(action_num/2),0+int(action_num/2))
		action = np.random.choice(action_mags,len(state))*action_disc # init. random
		actions.append(action)

		while term == False:
			i += 1

			# anneal action discretization
			action_disc /= i

			# epsilon-greedy action selection
			# switch = np.random.random_sample()
			# if switch > epsilon:
			action_prime = np.random.choice(action_mags,len(state))*action_disc
			# else:
			# 	# action_prime = -alpha_*np.ones(len(state)) # greedy action, maybe change this
			# 	action_prime = -action
			state_prime = np.add(state,action)

			samp = np.random.choice(Samples,batchsize)
			samp_prime = np.random.choice(Samples,batchsize)
			heuristic = [(heuristics(s_,dir_path),heuristics(s_p,dir_path)) for s_,s_p in zip(samp,samp_prime)]
			x_samp = [h[0] for h in heuristic]
			q_hat = np.mean(np.array([np.multiply(weights,np.multiply(state,np.mean(x_s,axis=1))) for x_s in x_samp]),axis=0)
			x_samp_prime = [h[1] for h in heuristic]
			q_hat_prime = np.mean(np.array([np.multiply(weights,np.multiply(state_prime,np.mean(x_sp,axis=1))) for x_sp in x_samp_prime]),axis=0)
			q_grad = np.mean(np.array([np.multiply(state,np.mean(x_s,axis=1)) for x_s in x_samp]),axis=0)

			reward = shortest_path(state,samp,dir_path) #change this to -(1-Accuracy) (i.e. minimize loss)

			if any(np.abs(state_prime)-5 > 5): # termination criteria
				reward -= 1
				weights_prime = np.add(weights,alpha_*np.multiply(np.subtract(reward,q_hat),q_grad))
				term = True
			else:
				# reward -= 0.1
				weights_prime = np.add(weights,alpha_*np.multiply(np.add(reward,np.subtract(gamma*q_hat_prime,q_hat)),q_grad))
			rewards.append(reward)			

			if any(np.abs(weights_prime)-5 > 5):
				term = True
			else:			
				weights = weights_prime

			if term	!= True:
				state = state_prime
				states.append(state)
				action = action_prime
				actions.append(action)

			if i > 100:
				term = True

		rewards_store[episode] = rewards
		states_store[episode] = states
		actions_store[episode] = actions
		weights_store[episode] = weights
		params = dict(episodes=episodes,samples=Samples,attributes=attributes,alpha=alpha,gamma=gamma,
				epsilon=epsilon,batchsize=batchsize,action_num=action_num,action_disc=action_disc)

		results = dict(rewards=rewards,states=states,actions=actions,weights=weights,params=params)
		with open('sarsa_batch_results_{}.pickle'.format(trial), 'wb') as f:
			# Pickle the results dictionary using the highest protocol available.
			pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

	return weights,rewards_store,states_store,actions_store,weights_store # all indexed/keyed by episode int

def semigradient_sarsa_continuous(episodes,Samples,attributes,alpha,gamma,epsilon,batchsize,action_num,action_disc,trial=0):  
	return 1

def main():

	tic = time.time()
	dir_path = os.path.dirname(os.path.realpath(__file__)) # where to write results pickle
	df = pd.read_csv('PercolationThresholds.txt',names=['Sample_Names','Percolation_Threshold'],sep='\t')
	Samples = [sample.strip('\'') for sample in df.Sample_Names]

	#algorithm parameters
	episodes = 100 # number of episodes
	attributes = 14
	alpha = 0.01 # initial step size for each episode
	gamma = 1 # undiscounted
	epsilon = 0.1
	batchsize = 10 # number of samples at each step
	action_num = 3 # dimension of action space
	action_disc = 1 # centered at 0, steps of this to either side
	trial = 1

	h_list = ['ArcLength', 'EuclideanDistance', 'MinWidth', 'meanWidth', 'inv_meanWidth', 'LongandThick', 'widestBottleneck',
				'Quino', 'Curvature', 'Straightness', 'Volume', 'AspectRatio', 'Poiseuille', 'Connectivity'][0:attributes]
	
	local_path = os.path.join(dir_path,'sarsa_batch_results_{}'.format(trial))
	if 'sarsa_batch_results_{}'.format(trial) in os.listdir(dir_path):
			rmtree('sarsa_batch_results_{}'.format(trial)) # only turn on if need to do again
	os.mkdir(local_path)
	os.chdir(local_path)
	weights,rewards_store,states_store,actions_store,weights_store = semigradient_sarsa_batch(episodes,Samples,attributes,alpha,gamma,epsilon,batchsize,action_num,action_disc,dir_path,trial)
	fig, ax = plt.subplots(2,1,figsize=(5,7)) 
	# ax.set_aspect('equal')

	attr_plot  = [states_store[i][-1] for i in states_store]
	a_store = pd.DataFrame(attr_plot,columns=h_list)
	w_store=pd.DataFrame.from_dict(weights_store,orient='index')
	w_store.columns = h_list
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
	os.chdir(dir_path)
	toc = time.time()
	print('runtime = ', toc - tic)

if __name__ == '__main__':
	main()

