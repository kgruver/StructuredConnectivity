import numpy as np
import numpy.random as random
from numpy import genfromtxt
from scipy import stats
from scipy import spatial
from scipy.spatial import distance
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.stats import percentileofscore


def comparedistances(csv1,csv2,sumofdistances):
    
    array1 = genfromtxt(csv1, delimiter=',', encoding='utf-8-sig')
    array2 = genfromtxt(csv2, delimiter=',', encoding='utf-8-sig')
    sumofdistancesperanimal = genfromtxt(sumofdistances, delimiter=',', encoding='utf-8-sig')

    #Assessing nearest neighbors of puncta from channel 1 to channel 2
    kdtree1 = spatial.KDTree(array2)
    neighbors_array1 = kdtree1.query(array1)
    neighbors_array1_transposed = np.transpose(neighbors_array1)
    
    #Assessing nearest neighbors of puncta from channel 2 to channel 1
    kdtree2 = spatial.KDTree(array1)
    neighbors_array2 = kdtree2.query(array2)
    neighbors_array2_transposed = np.transpose(neighbors_array2)
    
    #Combining the nearest neighbor distances for puncta across the two channels
    combined_array = np.concatenate((neighbors_array1_transposed, neighbors_array2_transposed), axis=0)
    
    #Dropping ID of puncta
    combined_distances = np.delete(combined_array, 1, 1)
    combined_distances_list = [item for sublist in combined_distances for item in sublist]
    
    #N.B. You can print out the nearest neighbor distances across both channels for one image in a stack by un-commenting the command below
    #print(combined_distances_list)

    #Calculating the percentile for number of puncta within X microns.
    #Range starts at 0 microns and ends at 200 microns, but can be edited.

    for i in range(0,200):
        percentile = percentileofscore(sumofdistancesperanimal, i)
        print(percentile)
        
    
### The above code generates percentiles of distances only across channels for one image in a stack. For data in figure, all the distances across stack positions were concatenated in a separate spreadsheet and the percentiles are measured for each animal.
### The percentiles are then averaged across all animals to get the final distribution. 

### In a separate command line, call the following function and replace csv1 and csv2 with the file paths of the .csv files included in this folder.

### For example:
csv1 = '/Animal_ID6/Animal_ID6_slice1_ch1_centroid_XYcoordinates_csv1.csv'
csv2 = '/Animal_ID6/Animal_ID6_slice1_ch2_centroid_XYcoordinates_csv2.csv'
sumofdistances = 'Distances_across_both_channels_Animal_ID6.csv'
comparedistances(csv1,csv2,sumofdistances)
 
