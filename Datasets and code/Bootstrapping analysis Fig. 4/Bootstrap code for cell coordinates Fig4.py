import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import random
import numpy.random as random
from numpy import genfromtxt
from scipy import stats
from scipy.stats import norm
from scipy.stats import sem
import csv
from scipy.stats import percentileofscore



# using SciPy stats models for linear regression

def Linear_regression_bootstrap_coordinates(axis1, axis2, axis, nZones, nShuffles):
    
    file = '3d coordinates_all75cells.csv'

    ###n4 = number of cells with n=4 zones
    n0 = 34
    n1 = 26
    n2 = 6
    n3 = 6
    n4 = 3

    ### Creating a pandas dataframe from .csv file
    pd_coordinates = pd.read_csv(file) 
    pd_coordinates_RC = pd_coordinates['Original_all_x']
    pd_coordinates_VD = pd_coordinates['Original_all_y']
    pd_coordinates_LM = pd_coordinates['Original_all_x']
    pd_coordinates_nZones = pd_coordinates['Original_coordinates_nZone']
    
    coordinates_RC = pd_coordinates_RC.to_numpy()
    coordinates_VD = pd_coordinates_VD.to_numpy()
    coordinates_LM = pd_coordinates_LM.to_numpy()
    
    x = coordinates_RC
    y = coordinates_VD
    z = coordinates_LM
    
    np_xy = np.dstack((x,y))
    np_xz = np.dstack((x,z))
    np_yz = np.dstack((y,z))


    ### Manually inputting coordinates in numpy array format 
    
    n0_x = np.array([12107,11923,11977,11925,12005,11921,11890,11752,12019,12022,12116,11926,12068,12040,11995,11995,11967,11995,11965,12056,11861,11844,11980,11716,11965,11851,11838,12008,12068,12046,11302,11365,11839,12006])
    n0_y = np.array([3378,3456,3476,3465,3537,3479,3445,3558,3463,3487,3465,3636,3510,3516,3585,3560,3516,3481,3491,3690,3628,3401,3440,3476,3725,3631,3645,3595,3619,3609,3845,3925,3755,3645])
    n0_z = np.array([4240,4240,4240,4240,4240,4240,4360,4360,4360,4360,4360,4360,4360,4360,4360,4540,4540,4540,4540,4540,4540,4540,4540,4660,4660,4660,4840,4840,4840,4840,4840,4840,4840,4960])
    
    n1_x = np.array([11751,11375,11715,11795,11789,11874,12003,11758,12135,12095,11912,12067,12059,11927,12040,12053,12058,12101,11777,11479,11485,12012,12045,11896,11862,12133])
    n1_y = np.array([3455,3707,3535,3465,3624,3545,3555,3450,3436,3544,3505,3560,3472,3562,3480,3454,3564,3647,3521,3904,3930,3761,3564,3656,3525,3581]) 
    n1_z = np.array([4660,4960,4960,4540,4540,4960,4400,4240,4240,4840,4540,4540,4240,4660,4360,4360,4540,4540,4660,4840,4840,4660,4960,4240,4540,4840])
    
    n2_x = np.array([11997,11878,11546,12105,11839,12080])
    n2_y = np.array([3535,3587,3889,3675,3743,3570]) 
    n2_z = np.array([4360,4960,4840,4960,4840,4840])
    
    n3_x = np.array([11846,11881,11798,11985,11920,11896])
    n3_y = np.array([3641,3624,3800,3479,3654,3615]) 
    n3_z = np.array([4360,4360,4840,4540,4660,4660])
    
    
    fourzone_x = np.array([11985,11761,11968])
    fourzone_y = np.array([3608,3760,3611])
    fourzone_z = np.array([4540,4960,4840])

    nOutliers = 0
    j = 0
    i = 0
    
    rng = np.random.default_rng()
    
    samples_of_34_nShuffles_times = []
    samples_of_26_nShuffles_times = []
    samples_of_6_nShuffles_times = []
    samples_of_6_nShuffles_times = []
    samples_of_3_nShuffles_times = []
    
    
    if nZones == "n4":
        if axis == "xy":

            fourzone_results = stats.linregress(fourzone_x,fourzone_y)

            fourzone_r2 = fourzone_results.rvalue**2
            fourzone_slope = fourzone_results.slope
            fourzone_yintercept = fourzone_results.intercept
            fourzone_pvalue = fourzone_results.pvalue

            fourzone_stats = ([fourzone_r2,fourzone_slope,fourzone_yintercept,fourzone_pvalue])


            for i in range(nShuffles):
                sample_of_3_i = pd.DataFrame()
                sample_of_3_i = pd_coordinates.sample(n4)
                sample_of_3_i_xy = sample_of_3_i[['Original_all_x','Original_all_y']].to_numpy()
                sample_of_3_i_xz = sample_of_3_i[['Original_all_x','Original_all_z']].to_numpy()
                sample_of_3_i_yz = sample_of_3_i[['Original_all_y','Original_all_z']].to_numpy()

                sample_of_3_i_x = sample_of_3_i_xy[:,0]
                sample_of_3_i_y = sample_of_3_i_xy[:,1]
                sample_of_3_i_z = sample_of_3_i_xz[:,1]

                sample_of_3_i_xy_linear_reg = stats.linregress(sample_of_3_i_x,sample_of_3_i_y)
                sample_of_3_i_xy_r = sample_of_3_i_xy_linear_reg.rvalue
                sample_of_3_i_xy_r2 = sample_of_3_i_xy_linear_reg.rvalue**2

                sample_of_3_i_xy_r_np = np.corrcoef(sample_of_3_i_x, sample_of_3_i_y)

                sample_of_3_i_xy_slope = sample_of_3_i_xy_linear_reg.slope
                sample_of_3_i_xy_yintercept = sample_of_3_i_xy_linear_reg.intercept
                sample_of_3_i_xy_pvalue = sample_of_3_i_xy_linear_reg.pvalue
                sample_of_3_i_xy_linear_reg_stats = ([sample_of_3_i_xy_r2,sample_of_3_i_xy_slope,sample_of_3_i_xy_yintercept,sample_of_3_i_xy_pvalue])

                samples_of_3_nShuffles_times += sample_of_3_i_xy_linear_reg_stats



                for nOutliers in range(len(sample_of_3_i_xy_r_np)):
                    if sample_of_3_i_xy_r2 >= fourzone_r2:
                        j = j + nOutliers
                        print(j)


        elif axis == "xz":

            fourzone_results = stats.linregress(fourzone_x,fourzone_z)

            fourzone_r2 = fourzone_results.rvalue**2
            fourzone_slope = fourzone_results.slope
            fourzone_yintercept = fourzone_results.intercept
            fourzone_pvalue = fourzone_results.pvalue

            fourzone_stats = ([fourzone_r2,fourzone_slope,fourzone_yintercept,fourzone_pvalue])


            for i in range(nShuffles):   

                sample_of_3_i = pd.DataFrame()
                sample_of_3_i = pd_coordinates.sample(n4)
                sample_of_3_i_xy = sample_of_3_i[['Original_all_x','Original_all_y']].to_numpy()
                sample_of_3_i_xz = sample_of_3_i[['Original_all_x','Original_all_z']].to_numpy()
                sample_of_3_i_yz = sample_of_3_i[['Original_all_y','Original_all_z']].to_numpy()

                sample_of_3_i_x = sample_of_3_i_xy[:,0]
                sample_of_3_i_y = sample_of_3_i_xy[:,1]
                sample_of_3_i_z = sample_of_3_i_xz[:,1]

                sample_of_3_i_xz_linear_reg = stats.linregress(sample_of_3_i_x,sample_of_3_i_z)
                sample_of_3_i_xz_r = sample_of_3_i_xz_linear_reg.rvalue
                sample_of_3_i_xz_r2 = sample_of_3_i_xz_linear_reg.rvalue**2

                sample_of_3_i_xz_r_np = np.corrcoef(sample_of_3_i_x, sample_of_3_i_z)

                sample_of_3_i_xz_slope = sample_of_3_i_xz_linear_reg.slope
                sample_of_3_i_xz_yintercept = sample_of_3_i_xz_linear_reg.intercept
                sample_of_3_i_xz_pvalue = sample_of_3_i_xz_linear_reg.pvalue
                sample_of_3_i_xz_linear_reg_stats = ([sample_of_3_i_xz_r2,sample_of_3_i_xz_slope,sample_of_3_i_xz_yintercept,sample_of_3_i_xz_pvalue])

                samples_of_3_nShuffles_times += sample_of_3_i_xz_linear_reg_stats


                for nOutliers in range(len(sample_of_3_i_xz_r_np)):
                    if sample_of_3_i_xz_r2 >= fourzone_r2:
                        j = j + nOutliers
                        print(j)


        else:
            axis == "yz"

            fourzone_results = stats.linregress(fourzone_y,fourzone_z)

            fourzone_r2 = fourzone_results.rvalue**2
            fourzone_slope = fourzone_results.slope
            fourzone_yintercept = fourzone_results.intercept
            fourzone_pvalue = fourzone_results.pvalue

            fourzone_stats = ([fourzone_r2,fourzone_slope,fourzone_yintercept,fourzone_pvalue])

            for i in range(nShuffles):   
                sample_of_3_i = pd.DataFrame()
                sample_of_3_i = pd_coordinates.sample(n4)
                sample_of_3_i_xy = sample_of_3_i[['Original_all_x','Original_all_y']].to_numpy()
                sample_of_3_i_xz = sample_of_3_i[['Original_all_x','Original_all_z']].to_numpy()
                sample_of_3_i_yz = sample_of_3_i[['Original_all_y','Original_all_z']].to_numpy()
                sample_of_3_i_x = sample_of_3_i_xy[:,0]
                sample_of_3_i_y = sample_of_3_i_xy[:,1]
                sample_of_3_i_z = sample_of_3_i_xz[:,1]

                sample_of_3_i_yz_linear_reg = stats.linregress(sample_of_3_i_y,sample_of_3_i_z)
                sample_of_3_i_yz_r = sample_of_3_i_yz_linear_reg.rvalue
                sample_of_3_i_yz_r2 = sample_of_3_i_yz_linear_reg.rvalue**2

                sample_of_3_i_yz_r_np = np.corrcoef(sample_of_3_i_y, sample_of_3_i_z)

                sample_of_3_i_yz_slope = sample_of_3_i_yz_linear_reg.slope
                sample_of_3_i_yz_yintercept = sample_of_3_i_yz_linear_reg.intercept
                sample_of_3_i_yz_pvalue = sample_of_3_i_yz_linear_reg.pvalue
                sample_of_3_i_yz_linear_reg_stats = ([sample_of_3_i_yz_r2,sample_of_3_i_yz_slope,sample_of_3_i_yz_yintercept,sample_of_3_i_yz_pvalue])

                samples_of_3_nShuffles_times += sample_of_3_i_yz_linear_reg_stats


                for nOutliers in range(len(sample_of_3_i_yz_r_np)):
                    if sample_of_3_i_yz_r2 >= fourzone_r2:
                        j = j + nOutliers
                        print(j)


        print("fourzone r2 value for "+str(axis)+" is: "+str(fourzone_r2))

        samples_of_3_nShuffles_times_reshaped = np.reshape(samples_of_3_nShuffles_times, (nShuffles,4))


        samplesof3_pvalue = j/nShuffles
        print("Pvalue of r2 happening this many times: "+str(samplesof3_pvalue))

        print(samples_of_3_nShuffles_times_reshaped)
        np.savetxt("stats_samples_of_3cells_n4_shuffled_n_times.csv",samples_of_3_nShuffles_times_reshaped,delimiter=",")

        
    elif nZones == "n3":
        if axis == "xy":

            n3_results = stats.linregress(n3_x,n3_y)

            n3_r2 = n3_results.rvalue**2
            n3_slope = n3_results.slope
            n3_yintercept = n3_results.intercept
            n3_pvalue = n3_results.pvalue

            n3_stats = ([n3_r2,n3_slope,n3_yintercept,n3_pvalue])


            for i in range(nShuffles):
                sample_of_6_i = pd.DataFrame()
                sample_of_6_i = pd_coordinates.sample(n3)
                sample_of_6_i_xy = sample_of_6_i[['Original_all_x','Original_all_y']].to_numpy()
                sample_of_6_i_xz = sample_of_6_i[['Original_all_x','Original_all_z']].to_numpy()
                sample_of_6_i_yz = sample_of_6_i[['Original_all_y','Original_all_z']].to_numpy()

                sample_of_6_i_x = sample_of_6_i_xy[:,0]
                sample_of_6_i_y = sample_of_6_i_xy[:,1]
                sample_of_6_i_z = sample_of_6_i_xz[:,1]

                sample_of_6_i_xy_linear_reg = stats.linregress(sample_of_6_i_x,sample_of_6_i_y)
                sample_of_6_i_xy_r = sample_of_6_i_xy_linear_reg.rvalue
                sample_of_6_i_xy_r2 = sample_of_6_i_xy_linear_reg.rvalue**2

                sample_of_6_i_xy_r_np = np.corrcoef(sample_of_6_i_x, sample_of_6_i_y)

                sample_of_6_i_xy_slope = sample_of_6_i_xy_linear_reg.slope
                sample_of_6_i_xy_yintercept = sample_of_6_i_xy_linear_reg.intercept
                sample_of_6_i_xy_pvalue = sample_of_6_i_xy_linear_reg.pvalue
                sample_of_6_i_xy_linear_reg_stats = ([sample_of_6_i_xy_r2,sample_of_6_i_xy_slope,sample_of_6_i_xy_yintercept,sample_of_6_i_xy_pvalue])

                samples_of_6_nShuffles_times += sample_of_6_i_xy_linear_reg_stats



                for nOutliers in range(len(sample_of_6_i_xy_r_np)):
                    if sample_of_6_i_xy_r2 >= n3_r2:
                        j = j + nOutliers
                        print(j)


        elif axis == "xz":

            n3_results = stats.linregress(n3_x,n3_z)

            n3_r2 = n3_results.rvalue**2
            n3_slope = n3_results.slope
            n3_yintercept = n3_results.intercept
            n3_pvalue = n3_results.pvalue

            n3_stats = ([n3_r2,n3_slope,n3_yintercept,n3_pvalue])


            for i in range(nShuffles):
                sample_of_6_i = pd.DataFrame()
                sample_of_6_i = pd_coordinates.sample(n3)
                sample_of_6_i_xy = sample_of_6_i[['Original_all_x','Original_all_y']].to_numpy()
                sample_of_6_i_xz = sample_of_6_i[['Original_all_x','Original_all_z']].to_numpy()
                sample_of_6_i_yz = sample_of_6_i[['Original_all_y','Original_all_z']].to_numpy()

                sample_of_6_i_x = sample_of_6_i_xy[:,0]
                sample_of_6_i_y = sample_of_6_i_xy[:,1]
                sample_of_6_i_z = sample_of_6_i_xz[:,1]

                sample_of_6_i_xz_linear_reg = stats.linregress(sample_of_6_i_x,sample_of_6_i_z)
                sample_of_6_i_xz_r = sample_of_6_i_xz_linear_reg.rvalue
                sample_of_6_i_xz_r2 = sample_of_6_i_xz_linear_reg.rvalue**2

                sample_of_6_i_xz_r_np = np.corrcoef(sample_of_6_i_x, sample_of_6_i_z)

                sample_of_6_i_xz_slope = sample_of_6_i_xz_linear_reg.slope
                sample_of_6_i_xz_yintercept = sample_of_6_i_xz_linear_reg.intercept
                sample_of_6_i_xz_pvalue = sample_of_6_i_xz_linear_reg.pvalue
                sample_of_6_i_xz_linear_reg_stats = ([sample_of_6_i_xz_r2,sample_of_6_i_xz_slope,sample_of_6_i_xz_yintercept,sample_of_6_i_xz_pvalue])

                samples_of_6_nShuffles_times += sample_of_6_i_xz_linear_reg_stats



                for nOutliers in range(len(sample_of_6_i_xz_r_np)):
                    if sample_of_6_i_xz_r2 >= n3_r2:
                        j = j + nOutliers
                        print(j)

        else:
            n3_results = stats.linregress(n3_y,n3_z)

            n3_r2 = n3_results.rvalue**2
            n3_slope = n3_results.slope
            n3_yintercept = n3_results.intercept
            n3_pvalue = n3_results.pvalue

            n3_stats = ([n3_r2,n3_slope,n3_yintercept,n3_pvalue])


            for i in range(nShuffles):
                sample_of_6_i = pd.DataFrame()
                sample_of_6_i = pd_coordinates.sample(n3)
                sample_of_6_i_xy = sample_of_6_i[['Original_all_x','Original_all_y']].to_numpy()
                sample_of_6_i_xz = sample_of_6_i[['Original_all_x','Original_all_z']].to_numpy()
                sample_of_6_i_yz = sample_of_6_i[['Original_all_y','Original_all_z']].to_numpy()

                sample_of_6_i_x = sample_of_6_i_xy[:,0]
                sample_of_6_i_y = sample_of_6_i_xy[:,1]
                sample_of_6_i_z = sample_of_6_i_xz[:,1]

                sample_of_6_i_yz_linear_reg = stats.linregress(sample_of_6_i_y,sample_of_6_i_z)
                sample_of_6_i_yz_r = sample_of_6_i_yz_linear_reg.rvalue
                sample_of_6_i_yz_r2 = sample_of_6_i_yz_linear_reg.rvalue**2

                sample_of_6_i_yz_r_np = np.corrcoef(sample_of_6_i_y, sample_of_6_i_z)

                sample_of_6_i_yz_slope = sample_of_6_i_yz_linear_reg.slope
                sample_of_6_i_yz_yintercept = sample_of_6_i_yz_linear_reg.intercept
                sample_of_6_i_yz_pvalue = sample_of_6_i_yz_linear_reg.pvalue
                sample_of_6_i_yz_linear_reg_stats = ([sample_of_6_i_yz_r2,sample_of_6_i_yz_slope,sample_of_6_i_yz_yintercept,sample_of_6_i_yz_pvalue])

                samples_of_6_nShuffles_times += sample_of_6_i_yz_linear_reg_stats



                for nOutliers in range(len(sample_of_6_i_yz_r_np)):
                    if sample_of_6_i_yz_r2 >= n3_r2:
                        j = j + nOutliers
                        print(j)


        print("n3 r2 value for "+str(axis)+" is: "+str(n3_r2))

        samples_of_6_nShuffles_times_reshaped = np.reshape(samples_of_6_nShuffles_times, (nShuffles,4))


        samplesof6_pvalue = j/nShuffles
        print("Pvalue of r2 happening this many times: "+str(samplesof6_pvalue))

        print(samples_of_6_nShuffles_times_reshaped)
        np.savetxt("stats_samples_of_6cells_n3_shuffled_n_times.csv",samples_of_6_nShuffles_times_reshaped,delimiter=",")

        
    elif nZones == "n2":
        if axis == "xy":

            n2_results = stats.linregress(n2_x,n2_y)

            n2_r2 = n2_results.rvalue**2
            n2_slope = n2_results.slope
            n2_yintercept = n2_results.intercept
            n2_pvalue = n2_results.pvalue

            n2_stats = ([n2_r2,n2_slope,n2_yintercept,n2_pvalue])


            for i in range(nShuffles):
                sample_of_6_i = pd.DataFrame()
                sample_of_6_i = pd_coordinates.sample(n2)
                sample_of_6_i_xy = sample_of_6_i[['Original_all_x','Original_all_y']].to_numpy()
                sample_of_6_i_xz = sample_of_6_i[['Original_all_x','Original_all_z']].to_numpy()
                sample_of_6_i_yz = sample_of_6_i[['Original_all_y','Original_all_z']].to_numpy()

                sample_of_6_i_x = sample_of_6_i_xy[:,0]
                sample_of_6_i_y = sample_of_6_i_xy[:,1]
                sample_of_6_i_z = sample_of_6_i_xz[:,1]

                sample_of_6_i_xy_linear_reg = stats.linregress(sample_of_6_i_x,sample_of_6_i_y)
                sample_of_6_i_xy_r = sample_of_6_i_xy_linear_reg.rvalue
                sample_of_6_i_xy_r2 = sample_of_6_i_xy_linear_reg.rvalue**2

                sample_of_6_i_xy_r_np = np.corrcoef(sample_of_6_i_x, sample_of_6_i_y)

                sample_of_6_i_xy_slope = sample_of_6_i_xy_linear_reg.slope
                sample_of_6_i_xy_yintercept = sample_of_6_i_xy_linear_reg.intercept
                sample_of_6_i_xy_pvalue = sample_of_6_i_xy_linear_reg.pvalue
                sample_of_6_i_xy_linear_reg_stats = ([sample_of_6_i_xy_r2,sample_of_6_i_xy_slope,sample_of_6_i_xy_yintercept,sample_of_6_i_xy_pvalue])

                samples_of_6_nShuffles_times += sample_of_6_i_xy_linear_reg_stats



                for nOutliers in range(len(sample_of_6_i_xy_r_np)):
                    if sample_of_6_i_xy_r2 >= n2_r2:
                        j = j + nOutliers
                        print(j)


        elif axis == "xz":

            n2_results = stats.linregress(n2_x,n2_z)

            n2_r2 = n2_results.rvalue**2
            n2_slope = n2_results.slope
            n2_yintercept = n2_results.intercept
            n2_pvalue = n2_results.pvalue

            n2_stats = ([n2_r2,n2_slope,n2_yintercept,n2_pvalue])


            for i in range(nShuffles):
                sample_of_6_i = pd.DataFrame()
                sample_of_6_i = pd_coordinates.sample(n2)
                sample_of_6_i_xy = sample_of_6_i[['Original_all_x','Original_all_y']].to_numpy()
                sample_of_6_i_xz = sample_of_6_i[['Original_all_x','Original_all_z']].to_numpy()
                sample_of_6_i_yz = sample_of_6_i[['Original_all_y','Original_all_z']].to_numpy()

                sample_of_6_i_x = sample_of_6_i_xy[:,0]
                sample_of_6_i_y = sample_of_6_i_xy[:,1]
                sample_of_6_i_z = sample_of_6_i_xz[:,1]

                sample_of_6_i_xz_linear_reg = stats.linregress(sample_of_6_i_x,sample_of_6_i_z)
                sample_of_6_i_xz_r = sample_of_6_i_xz_linear_reg.rvalue
                sample_of_6_i_xz_r2 = sample_of_6_i_xz_linear_reg.rvalue**2

                sample_of_6_i_xz_r_np = np.corrcoef(sample_of_6_i_x, sample_of_6_i_z)

                sample_of_6_i_xz_slope = sample_of_6_i_xz_linear_reg.slope
                sample_of_6_i_xz_yintercept = sample_of_6_i_xz_linear_reg.intercept
                sample_of_6_i_xz_pvalue = sample_of_6_i_xz_linear_reg.pvalue
                sample_of_6_i_xz_linear_reg_stats = ([sample_of_6_i_xz_r2,sample_of_6_i_xz_slope,sample_of_6_i_xz_yintercept,sample_of_6_i_xz_pvalue])

                samples_of_6_nShuffles_times += sample_of_6_i_xz_linear_reg_stats



                for nOutliers in range(len(sample_of_6_i_xz_r_np)):
                    if sample_of_6_i_xz_r2 >= n2_r2:
                        j = j + nOutliers
                        print(j)

        else:
            n2_results = stats.linregress(n2_y,n2_z)

            n2_r2 = n2_results.rvalue**2
            n2_slope = n2_results.slope
            n2_yintercept = n2_results.intercept
            n2_pvalue = n2_results.pvalue

            n2_stats = ([n2_r2,n2_slope,n2_yintercept,n2_pvalue])


            for i in range(nShuffles):
                sample_of_6_i = pd.DataFrame()
                sample_of_6_i = pd_coordinates.sample(n2)
                sample_of_6_i_xy = sample_of_6_i[['Original_all_x','Original_all_y']].to_numpy()
                sample_of_6_i_xz = sample_of_6_i[['Original_all_x','Original_all_z']].to_numpy()
                sample_of_6_i_yz = sample_of_6_i[['Original_all_y','Original_all_z']].to_numpy()

                sample_of_6_i_x = sample_of_6_i_xy[:,0]
                sample_of_6_i_y = sample_of_6_i_xy[:,1]
                sample_of_6_i_z = sample_of_6_i_xz[:,1]

                sample_of_6_i_yz_linear_reg = stats.linregress(sample_of_6_i_y,sample_of_6_i_z)
                sample_of_6_i_yz_r = sample_of_6_i_yz_linear_reg.rvalue
                sample_of_6_i_yz_r2 = sample_of_6_i_yz_linear_reg.rvalue**2

                sample_of_6_i_yz_r_np = np.corrcoef(sample_of_6_i_y, sample_of_6_i_z)

                sample_of_6_i_yz_slope = sample_of_6_i_yz_linear_reg.slope
                sample_of_6_i_yz_yintercept = sample_of_6_i_yz_linear_reg.intercept
                sample_of_6_i_yz_pvalue = sample_of_6_i_yz_linear_reg.pvalue
                sample_of_6_i_yz_linear_reg_stats = ([sample_of_6_i_yz_r2,sample_of_6_i_yz_slope,sample_of_6_i_yz_yintercept,sample_of_6_i_yz_pvalue])

                samples_of_6_nShuffles_times += sample_of_6_i_yz_linear_reg_stats



                for nOutliers in range(len(sample_of_6_i_yz_r_np)):
                    if sample_of_6_i_yz_r2 >= n2_r2:
                        j = j + nOutliers
                        print(j)


        print("n2 r2 value for "+str(axis)+" is: "+str(n2_r2))

        samples_of_6_nShuffles_times_reshaped = np.reshape(samples_of_6_nShuffles_times, (nShuffles,4))


        samplesof6_pvalue = j/nShuffles
        print("Pvalue of r2 happening this many times: "+str(samplesof6_pvalue))

        print(samples_of_6_nShuffles_times_reshaped)
        np.savetxt("stats_samples_of_6cells_n2_shuffled_n_times.csv",samples_of_6_nShuffles_times_reshaped,delimiter=",")

    elif nZones == "n1":
        if axis == "xy":

            n1_results = stats.linregress(n1_x,n1_y)

            n1_r2 = n1_results.rvalue**2
            n1_slope = n1_results.slope
            n1_yintercept = n1_results.intercept
            n1_pvalue = n1_results.pvalue

            n1_stats = ([n1_r2,n1_slope,n1_yintercept,n1_pvalue])


            for i in range(nShuffles):
                sample_of_26_i = pd.DataFrame()
                sample_of_26_i = pd_coordinates.sample(n1)
                sample_of_26_i_xy = sample_of_26_i[['Original_all_x','Original_all_y']].to_numpy()
                sample_of_26_i_xz = sample_of_26_i[['Original_all_x','Original_all_z']].to_numpy()
                sample_of_26_i_yz = sample_of_26_i[['Original_all_y','Original_all_z']].to_numpy()

                sample_of_26_i_x = sample_of_26_i_xy[:,0]
                sample_of_26_i_y = sample_of_26_i_xy[:,1]
                sample_of_26_i_z = sample_of_26_i_xz[:,1]

                sample_of_26_i_xy_linear_reg = stats.linregress(sample_of_26_i_x,sample_of_26_i_y)
                sample_of_26_i_xy_r = sample_of_26_i_xy_linear_reg.rvalue
                sample_of_26_i_xy_r2 = sample_of_26_i_xy_linear_reg.rvalue**2

                sample_of_26_i_xy_r_np = np.corrcoef(sample_of_26_i_x, sample_of_26_i_y)

                sample_of_26_i_xy_slope = sample_of_26_i_xy_linear_reg.slope
                sample_of_26_i_xy_yintercept = sample_of_26_i_xy_linear_reg.intercept
                sample_of_26_i_xy_pvalue = sample_of_26_i_xy_linear_reg.pvalue
                sample_of_26_i_xy_linear_reg_stats = ([sample_of_26_i_xy_r2,sample_of_26_i_xy_slope,sample_of_26_i_xy_yintercept,sample_of_26_i_xy_pvalue])

                samples_of_26_nShuffles_times += sample_of_26_i_xy_linear_reg_stats



                for nOutliers in range(len(sample_of_26_i_xy_r_np)):
                    if sample_of_26_i_xy_r2 >= n1_r2:
                        j = j + nOutliers
                        print(j)


        elif axis == "xz":

            n1_results = stats.linregress(n1_x,n1_z)

            n1_r2 = n1_results.rvalue**2
            n1_slope = n1_results.slope
            n1_yintercept = n1_results.intercept
            n1_pvalue = n1_results.pvalue

            n1_stats = ([n1_r2,n1_slope,n1_yintercept,n1_pvalue])


            for i in range (nShuffles):
                sample_of_26_i = pd.DataFrame()
                sample_of_26_i = pd_coordinates.sample(n1)
                sample_of_26_i_xy = sample_of_26_i[['Original_all_x','Original_all_y']].to_numpy()
                sample_of_26_i_xz = sample_of_26_i[['Original_all_x','Original_all_z']].to_numpy()
                sample_of_26_i_yz = sample_of_26_i[['Original_all_y','Original_all_z']].to_numpy()

                sample_of_26_i_x = sample_of_26_i_xy[:,0]
                sample_of_26_i_y = sample_of_26_i_xy[:,1]
                sample_of_26_i_z = sample_of_26_i_xz[:,1]

                sample_of_26_i_xz_linear_reg = stats.linregress(sample_of_26_i_x,sample_of_26_i_z)
                sample_of_26_i_xz_r = sample_of_26_i_xz_linear_reg.rvalue
                sample_of_26_i_xz_r2 = sample_of_26_i_xz_linear_reg.rvalue**2

                sample_of_26_i_xz_r_np = np.corrcoef(sample_of_26_i_x, sample_of_26_i_z)

                sample_of_26_i_xz_slope = sample_of_26_i_xz_linear_reg.slope
                sample_of_26_i_xz_yintercept = sample_of_26_i_xz_linear_reg.intercept
                sample_of_26_i_xz_pvalue = sample_of_26_i_xz_linear_reg.pvalue
                sample_of_26_i_xz_linear_reg_stats = ([sample_of_26_i_xz_r2,sample_of_26_i_xz_slope,sample_of_26_i_xz_yintercept,sample_of_26_i_xz_pvalue])

                samples_of_26_nShuffles_times += sample_of_26_i_xz_linear_reg_stats



                for nOutliers in range(len(sample_of_26_i_xz_r_np)):
                    if sample_of_26_i_xz_r2 >= n1_r2:
                        j = j + nOutliers
                        print(j)

        else:
            n1_results = stats.linregress(n1_y,n1_z)

            n1_r2 = n1_results.rvalue**2
            n1_slope = n1_results.slope
            n1_yintercept = n1_results.intercept
            n1_pvalue = n1_results.pvalue

            n1_stats = ([n1_r2,n1_slope,n1_yintercept,n1_pvalue])


            for i in range(nShuffles):
                sample_of_26_i = pd.DataFrame()
                sample_of_26_i = pd_coordinates.sample(n1)
                sample_of_26_i_xy = sample_of_26_i[['Original_all_x','Original_all_y']].to_numpy()
                sample_of_26_i_xz = sample_of_26_i[['Original_all_x','Original_all_z']].to_numpy()
                sample_of_26_i_yz = sample_of_26_i[['Original_all_y','Original_all_z']].to_numpy()

                sample_of_26_i_x = sample_of_26_i_xy[:,0]
                sample_of_26_i_y = sample_of_26_i_xy[:,1]
                sample_of_26_i_z = sample_of_26_i_xz[:,1]
                sample_of_26_i_yz_linear_reg = stats.linregress(sample_of_26_i_y,sample_of_26_i_z)
                sample_of_26_i_yz_r = sample_of_26_i_yz_linear_reg.rvalue
                sample_of_26_i_yz_r2 = sample_of_26_i_yz_linear_reg.rvalue**2

                sample_of_26_i_yz_r_np = np.corrcoef(sample_of_26_i_y, sample_of_26_i_z)

                sample_of_26_i_yz_slope = sample_of_26_i_yz_linear_reg.slope
                sample_of_26_i_yz_yintercept = sample_of_26_i_yz_linear_reg.intercept
                sample_of_26_i_yz_pvalue = sample_of_26_i_yz_linear_reg.pvalue
                sample_of_26_i_yz_linear_reg_stats = ([sample_of_26_i_yz_r2,sample_of_26_i_yz_slope,sample_of_26_i_yz_yintercept,sample_of_26_i_yz_pvalue])

                samples_of_26_nShuffles_times += sample_of_26_i_yz_linear_reg_stats



                for nOutliers in range(len(sample_of_26_i_yz_r_np)):
                     if sample_of_26_i_yz_r2 >= n1_r2:
                        j = j + nOutliers
                        print(j)


        print("n1 r2 value for "+str(axis)+" is: "+str(n1_r2))

        samples_of_26_nShuffles_times_reshaped = np.reshape(samples_of_26_nShuffles_times, (nShuffles,4))


        samplesof26_pvalue = j/nShuffles
        print("Pvalue of r2 happening this many times: "+str(samplesof26_pvalue))

        print(samples_of_26_nShuffles_times_reshaped)
        np.savetxt("stats_samples_of_26cells_n1_shuffled_n_times.csv",samples_of_26_nShuffles_times_reshaped,delimiter=",")

        
        
    else:
        if axis == "xy":

            n0_results = stats.linregress(n0_x,n0_y)

            n0_r2 = n0_results.rvalue**2
            n0_slope = n0_results.slope
            n0_yintercept = n0_results.intercept
            n0_pvalue = n0_results.pvalue

            n0_stats = ([n0_r2,n0_slope,n0_yintercept,n0_pvalue])


            for i in range(nShuffles):
                    sample_of_34_i = pd.DataFrame()
                    sample_of_34_i = pd_coordinates.sample(n0)
                    sample_of_34_i_xy = sample_of_34_i[['Original_all_x','Original_all_y']].to_numpy()
                    sample_of_34_i_xz = sample_of_34_i[['Original_all_x','Original_all_z']].to_numpy()
                    sample_of_34_i_yz = sample_of_34_i[['Original_all_y','Original_all_z']].to_numpy()

                    sample_of_34_i_x = sample_of_34_i_xy[:,0]
                    sample_of_34_i_y = sample_of_34_i_xy[:,1]
                    sample_of_34_i_z = sample_of_34_i_xz[:,1]

                    sample_of_34_i_xy_linear_reg = stats.linregress(sample_of_34_i_x,sample_of_34_i_y)
                    sample_of_34_i_xy_r = sample_of_34_i_xy_linear_reg.rvalue
                    sample_of_34_i_xy_r2 = sample_of_34_i_xy_linear_reg.rvalue**2

                    sample_of_34_i_xy_r_np = np.corrcoef(sample_of_34_i_x, sample_of_34_i_y)

                    sample_of_34_i_xy_slope = sample_of_34_i_xy_linear_reg.slope
                    sample_of_34_i_xy_yintercept = sample_of_34_i_xy_linear_reg.intercept
                    sample_of_34_i_xy_pvalue = sample_of_34_i_xy_linear_reg.pvalue
                    sample_of_34_i_xy_linear_reg_stats = ([sample_of_34_i_xy_r2,sample_of_34_i_xy_slope,sample_of_34_i_xy_yintercept,sample_of_34_i_xy_pvalue])

                    samples_of_34_nShuffles_times += sample_of_34_i_xy_linear_reg_stats



                    for nOutliers in range(len(sample_of_34_i_xy_r_np)):
                        if sample_of_34_i_xy_r2 >= n0_r2:
                            j = j + nOutliers
                            print(j)


        elif axis == "xz":

            n0_results = stats.linregress(n0_x,n0_z)

            n0_r2 = n0_results.rvalue**2
            n0_slope = n0_results.slope
            n0_yintercept = n0_results.intercept
            n0_pvalue = n0_results.pvalue

            n0_stats = ([n0_r2,n0_slope,n0_yintercept,n0_pvalue])


            for i in range(nShuffles):
                    sample_of_34_i = pd.DataFrame()
                    sample_of_34_i = pd_coordinates.sample(34)
                    sample_of_34_i_xy = sample_of_34_i[['Original_all_x','Original_all_y']].to_numpy()
                    sample_of_34_i_xz = sample_of_34_i[['Original_all_x','Original_all_z']].to_numpy()
                    sample_of_34_i_yz = sample_of_34_i[['Original_all_y','Original_all_z']].to_numpy()

                    sample_of_34_i_x = sample_of_34_i_xy[:,0]
                    sample_of_34_i_y = sample_of_34_i_xy[:,1]
                    sample_of_34_i_z = sample_of_34_i_xz[:,1]

                    sample_of_34_i_xz_linear_reg = stats.linregress(sample_of_34_i_x,sample_of_34_i_z)
                    sample_of_34_i_xz_r = sample_of_34_i_xz_linear_reg.rvalue
                    sample_of_34_i_xz_r2 = sample_of_34_i_xz_linear_reg.rvalue**2

                    sample_of_34_i_xz_r_np = np.corrcoef(sample_of_34_i_x, sample_of_34_i_z)

                    sample_of_34_i_xz_slope = sample_of_34_i_xz_linear_reg.slope
                    sample_of_34_i_xz_yintercept = sample_of_34_i_xz_linear_reg.intercept
                    sample_of_34_i_xz_pvalue = sample_of_34_i_xz_linear_reg.pvalue
                    sample_of_34_i_xz_linear_reg_stats = ([sample_of_34_i_xz_r2,sample_of_34_i_xz_slope,sample_of_34_i_xz_yintercept,sample_of_34_i_xz_pvalue])

                    samples_of_34_nShuffles_times += sample_of_34_i_xz_linear_reg_stats



            for nOutliers in range(len(sample_of_34_i_xz_r_np)):
                 if sample_of_34_i_xz_r2 >= n0_r2:
                    j = j + nOutliers
                    print(j)

        else:
            n0_results = stats.linregress(n0_y,n0_z)

            n0_r2 = n0_results.rvalue**2
            n0_slope = n0_results.slope
            n0_yintercept = n0_results.intercept
            n0_pvalue = n0_results.pvalue

            n0_stats = ([n0_r2,n0_slope,n0_yintercept,n0_pvalue])


            for i in range(nShuffles):
                sample_of_34_i = pd.DataFrame()
                sample_of_34_i = pd_coordinates.sample(n0)
                sample_of_34_i_xy = sample_of_34_i[['Original_all_x','Original_all_y']].to_numpy()
                sample_of_34_i_xz = sample_of_34_i[['Original_all_x','Original_all_z']].to_numpy()
                sample_of_34_i_yz = sample_of_34_i[['Original_all_y','Original_all_z']].to_numpy()

                sample_of_34_i_x = sample_of_34_i_xy[:,0]
                sample_of_34_i_y = sample_of_34_i_xy[:,1]
                sample_of_34_i_z = sample_of_34_i_xz[:,1]

                sample_of_34_i_yz_linear_reg = stats.linregress(sample_of_34_i_y,sample_of_34_i_z)
                sample_of_34_i_yz_r = sample_of_34_i_yz_linear_reg.rvalue
                sample_of_34_i_yz_r2 = sample_of_34_i_yz_linear_reg.rvalue**2

                sample_of_34_i_yz_r_np = np.corrcoef(sample_of_34_i_y, sample_of_34_i_z)

                sample_of_34_i_yz_slope = sample_of_34_i_yz_linear_reg.slope
                sample_of_34_i_yz_yintercept = sample_of_34_i_yz_linear_reg.intercept
                sample_of_34_i_yz_pvalue = sample_of_34_i_yz_linear_reg.pvalue
                sample_of_34_i_yz_linear_reg_stats = ([sample_of_34_i_yz_r2,sample_of_34_i_yz_slope,sample_of_34_i_yz_yintercept,sample_of_34_i_yz_pvalue])

                samples_of_34_nShuffles_times += sample_of_34_i_yz_linear_reg_stats

                for nOutliers in range(len(sample_of_34_i_yz_r_np)):
                    if sample_of_34_i_yz_r2 >= n0_r2:
                        j = j + nOutliers
                        print(j)


        print("n0 r2 value for "+str(axis)+" is: "+str(n0_r2))

        samples_of_34_nShuffles_times_reshaped = np.reshape(samples_of_34_nShuffles_times, (nShuffles,4))


        samplesof34_pvalue = j/nShuffles
        print("Pvalue of r2 happening this many times: "+str(samplesof34_pvalue))

        print(samples_of_34_nShuffles_times_reshaped)
        np.savetxt("stats_samples_of_34cells_n0_shuffled_n_times.csv",samples_of_34_nShuffles_times_reshaped,delimiter=",")
