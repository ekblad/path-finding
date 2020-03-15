import h5py
import pandas as pd
import time
#column 1: Sample Name
#column 2: Percolation Threshold: indicates level of preferentiality of the flow, lower = uniform flow, higher = preferential flow
#column 2: node ID
#column 3: x-coordinate
#column 4: y-coordinate
#column 5: Edge ID - corresponds to the EdgeCoordinates.hdf5 file for a particular sample
#column 6: Arc Length
#column 7: Mean Width
#column 8: "Long and Thick": Arc Length/Mean Width
#column 9: "Curvature": Euclidean Distance^2/Arc Length
#column 10: "Connectivity": node degree
tic = time.time()
df = pd.read_csv('PercolationThresholds.txt',names = ['Sample_Names','Percolation_Threshold'], sep ='\t')
everything = pd.DataFrame(columns = ['SampleName','PercolationThreshold','nodeID','x_coordinate','y_coordinate','edgeID','ArcLength','MeanWidth','LongandThick','Curvature','Connectivity'])
dir_path = r'C:\Users\zkanavas\OneDrive\Documents\Previous_Semesters\Winter_2019\ECI_273\Homework\Term_Project\All_Samples_001'
Samples = [string.split("'")[1] for string in df.Sample_Names.to_list()]
for sample in Samples:
    print(sample)
    with h5py.File(dir_path+'/Sample_A/'+sample+'/'+'sparselist.hdf5','r') as f:
        sparselist = f.get('default')[()]
    for node in range(len(sparselist)): #range(0,len(sparselist),2)?
        everything = everything.append({'SampleName':sample,'PercolationThreshold':df.loc[df.Sample_Names=="'"+sample+"'",'Percolation_Threshold'].values[0],
        'nodeID':sparselist[node,0],'x_coordinate':sparselist[node,1],'y_coordinate':sparselist[node,2],'edgeID':sparselist[node,11],'ArcLength':sparselist[node,5],
        'MeanWidth':sparselist[node,7],'LongandThick':sparselist[node,5]/sparselist[node,7],'Curvature':sparselist[node,4]**2/sparselist[node,5],
        'Connectivity':sparselist[node,6]},ignore_index=True)
everything.to_csv('heuristic_info_all_samples.csv',index=False)
toc = time.time()
print('runtime = ', toc-tic)