import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import xarray # will be needed when panels in pandas gets deprecated


# TO DO: (OPTIONAL) Change code so that both .asc as .xlsx files 
# (+ provided grid info as parameter) can be read into function

def lookup_value(grid_file, input_file, sample, code_geol=None, average=True):
    """Lookup value of borehole value in grid file based 
    on coordinates of requested borehole sample in input file"""
    # Open file
    f = open(grid_file)
    grid = f.readlines()
    
    # Allocate grid info (Surfer grid info) and grid data (actual values)
    grid_info = grid[0:6]
    grid_data = grid[6:]
    
    # Put grid_info in dictionary
    grid_info_dict = {}

    for line in grid_info:
        key, value = line.split()
        grid_info_dict[key] = float(value)
    
    # Put grid_data in numpy matrix
    data = []

    for line in grid_data:
        data.append(line.split())
    
    data_array = np.array(data, dtype='float')    
    
    # Get grain size class from grid_file name
    regex = re.compile("z_\d+")
    grain_size_class = regex.search(grid_file).group()
    
    # TO DO: select sheet_name based on grid_file's name string
    if input_file.endswith(".csv"):
        lookup_values = pd.read_csv(input_file, sep=";", index_col="hole_id")
    else:
        lookup_values = pd.read_excel(input_file, index_col="hole_id")
    
    lon, lat = list(lookup_values.loc[sample, ["lat", "lon"]])
    grain_size = lookup_values.loc[sample, grain_size_class]
    
    # Adjust starting coordinates for the fact that Surfer uses 
    # half a cellsize offset in the xllcorner and yllcorner values
    start_lon = grid_info_dict["yllcorner"] + (grid_info_dict["cellsize"] / 2)
    start_lat = grid_info_dict["xllcorner"] + (grid_info_dict["cellsize"] / 2)
    
    # Used formula :
    # (requested coordinate - corrected starting coordinate) / cellsize = n
    # n  can be translated to the index in the grid data matrix
    y_actual = (lon - start_lon) / grid_info_dict["cellsize"] + 1
    x_actual = (lat - start_lat) / grid_info_dict["cellsize"]
    
    y1 = int(np.floor(y_actual))
    y2 = int(np.ceil(y_actual))

    x1 = int(np.floor(x_actual))
    x2 = int(np.ceil(x_actual))
    
    
#     y1 = int(np.floor((lon - start_lon) / grid_info_dict["cellsize"]) + 1)
#     y2 = int(np.ceil((lon - start_lon) / grid_info_dict["cellsize"]) + 1)
#     x1 = int(np.floor((lat - start_lat) / grid_info_dict["cellsize"]))
#     x2 = int(np.ceil((lat - start_lat) / grid_info_dict["cellsize"]))
    # Correction for index out of bounds error
    if x2 > grid_info_dict["ncols"] - 1:
        x2 -= 1
    
    # Calculate closest distance from actual coordinates to grid point 
    # coordinates
    # Choose minimal or maximal coordinates if on edge of grid
    
    y1_distance = np.abs(y1 - y_actual) 
    y2_distance = np.abs(y2 - y_actual)
    
    x1_distance = np.abs(x1 - x_actual)
    x2_distance = np.abs(x2 - x_actual)
    
    # Get coordinates with minimal distance
    
    if y1_distance < y2_distance:
        y_grid = y1
    else:
        y_grid = y2
    
    if x1_distance < x2_distance:
        x_grid = x1
    else:
        x_grid = x2
    
    # TO DO: Better to use closest point as coordinate to requested coordinate 
    # than taking the average of 4 nearest points
    
    
    # Create coordinate pairs for four closest points in the grid file
    if average == True:
        lookup_couples = []
        
        for x in [x1, x2]:
            for y in [y1, y2]:
                lookup_couples.append([int(grid_info_dict["nrows"]) - y, x])
        lookup_couples = np.array(lookup_couples)

        # Lookup values of the closest points in the grid file
        lookup_values_val = []

        for couple in lookup_couples:
            lookup_values_val.append(data_array[couple[0], couple[1]])

        # Take the mean of the found values
        mean_val = np.mean(lookup_values_val)
    else:
        print(y_grid, x_grid)
        mean_val = data_array[int(grid_info_dict["nrows"]) - y_grid, x_grid]
    
    f.close()
    
    return(mean_val, grain_size)


def cross_validation(rootDir):

    #quarry = "Berg"
    #code_geol = "GZ"
    #n_test = "1"

    results_test = {}

    n_files = 0
    

    for dirName, subdirList, fileList in os.walk(rootDir):

        if "Berg" in dirName:
            #print(dirName)
            quarry = "Berg"
        elif "MHZ" in dirName:
            #print(dirName)
            quarry = "MHZ"
            #results_geol = {}
        else:
            pass

        if "TZ" in dirName:
            code_geol = "TZ"
            print(code_geol)
        elif "IZ" in dirName:
            code_geol = "IZ"
            print(code_geol)
        elif "GZ" in dirName:
            code_geol="GZ"
            print(code_geol)
        else:
            pass
        
        #print(f"{quarry} {code_geol}")
        
        results_grain_size = {}

        for file in fileList:

            # New approach file syestem
            #if int(file[-5]) in [1, 2, 3, 4, 5]:
            #    n_test = file[-5]

            # Classic approach file system
            if int(dirName[-1]) in [1, 2, 3, 4, 5]:
                n_test = dirName[-1]
                #n_test_old = n_test
                
                inputfile = f"../_CROSS_VALIDATION_clr/{quarry}/{code_geol}/{quarry}_{code_geol}_{n_test}_test.csv"
                if inputfile.endswith(".csv"):
                    boreholes = list(pd.read_csv(inputfile, 
                                                 sep=";", 
                                                 index_col="hole_id").index)
                else:
                    boreholes = list(pd.read_excel(inputfile,  
                                                   index_col="hole_id").index)
                #if n_test_old != n_test:
                #    results_test = {}
                
            if file.endswith(".asc"):
                gridfile = file
                n_files +=1
                
                print(gridfile, inputfile)

                # Get grain size label
                regex = "z_\d+"
                grain_size = re.search(regex, file).group()

                # Get PC component label
                #regex1 = "\dcomp"
                #comp = re.search(regex1, file).group()

                # Get train label
                n_train = dirName[-1]

                results_borehole = {}

                # Loop through all test boreholes
                for borehole in boreholes:
                    results_borehole[borehole] = lookup_value(f"{dirName}/{gridfile}", 
                                                                 inputfile, 
                                                                 borehole, 
                                                                 average=False)
                    
                lookup_df = pd.DataFrame.from_dict(results_borehole, 
                                                   orient='index')
                lookup_df.columns = ["Grid", "Actual"]
                lookup_df["Diff"] = lookup_df["Grid"] - lookup_df["Actual"]
                
                
                results_grain_size[grain_size] = lookup_df
        try:
            results_test[n_train] = results_grain_size
        except:
            pass


    print("Number of files processed:", n_files)
    
    return(results_test)


def calculate_mse_new(CV_results):
    """Calculate Mean Squared Error based on 
    cross validation (CV) results"""

    mse_quarry = {}

    for quarry, geol_data in CV_results.items():
        mse_geol = {}

        for geol, comp_data in geol_data.items():
            mse_comp = {}

            for comp, train_data in comp_data.items():
                mse_train = {}

                for train, grain_size_data in train_data.items():
                    mse_grain_size = {}

                    for grain_size, data in grain_size_data.items():
                        # MSE calculation
                        mse_result = ((data["Actual"] - data["Grid"]) ** 2).mean(axis=0)
                        mse_grain_size[grain_size] = mse_result

                    mse_train[train] = mse_grain_size

                mse_comp[comp] = mse_train

            mse_geol[geol] = mse_comp

        mse_quarry[quarry] = mse_geol
        
    return mse_quarry


def calculate_mse_classic(CV_results):
    """Calculate Mean Squared Error based on 
    cross validation (CV) results"""

    mse_quarry = {}

    for quarry, geol_data in CV_results.items():
        mse_geol = {}

        for geol, train_data in geol_data.items():
            mse_train = {}

            for train, grain_size_data in train_data.items():
                mse_grain_size = {}

                for grain_size, data in grain_size_data.items():
                    # MSE calculation
                    mse_result = ((data["Actual"] - data["Grid"]) ** 2).mean(axis=0)
                    mse_grain_size[grain_size] = mse_result

                mse_train[train] = mse_grain_size

            mse_geol[geol] = mse_train

        mse_quarry[quarry] = mse_geol
        
    return mse_quarry


def average_mse_results_CVfolds_new(mse_results, quarry, code_geol, n_comp):

    averaged_mse_results = {}

    for train, grain_size_data in mse_results[quarry][code_geol][n_comp].items():
        values = []
        for grain_size, data in grain_size_data.items():
            values.append(data)
        averaged_mse_results[train] = np.mean(values)
    return(averaged_mse_results)


def average_mse_results_CVfolds_classic(mse_results, quarry, code_geol):

    averaged_mse_results = {}

    for train, grain_size_data in mse_results[quarry][code_geol].items():
        values = []
        for grain_size, data in grain_size_data.items():
            values.append(data)
        averaged_mse_results[train] = np.mean(values)
    return(averaged_mse_results)


def average_mse_results_n_comp(averaged_mse_results):
    
    values = []

    for key, value in averaged_mse_results.items():
        values.append(value)
        
    return(np.mean(values))