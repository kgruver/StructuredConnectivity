import numpy as np
import numpy.random as random
from numpy import genfromtxt

from scipy import stats
from scipy import spatial
from scipy.spatial import distance
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.stats import percentileofscore


def comparedistances(csv1,csv2):
    
    array1 = genfromtxt(csv1, delimiter=',')
    array2 = genfromtxt(csv2, delimiter=',')

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
    
    #N.B. You can print out the nearest neighbor distances across both channels by un-commenting the command below
    #print(combined_distances)
    
    #Calculating the percentile for number of puncta within X microns.
    #Range starts at 0 microns and ends at 100 microns, but can be edited.
    
    for i in range(0,100):
        percentile = percentileofscore(combined_distances, i)
        print(percentile)
        

### This is only for one image in the stack. All the distances are concatenated in a separate spreadsheet and the percentiles are measured for each animal.



### In a separate command line, call the following function and replace csv1 and csv2 with the file paths of the .csv files included in this folder.

#comparedistances(csv1,csv2)
 
