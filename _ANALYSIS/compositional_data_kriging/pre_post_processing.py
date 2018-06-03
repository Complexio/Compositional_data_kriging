import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import xarray # will be needed when panels in pandas gets deprecated


def pre_processing(file, save_data=False, save_name=None, 
                   column_range=["z_1000", "z_710", "z_500", "z_355", "z_250", 
                   "z_180", "z_125", "z_90", "z_63", "z_0"]):
    """Pre-processing module of new approach for interpolation 
    of compositional data by ordinary kriging

    Parameters:

    Returns:

    """
    
    # -------------------------
    # DATA LOADING AND CLEANING
    # -------------------------
    
    xlsx_file = pd.ExcelFile(file)
    print(xlsx_file.sheet_names)
    
    # Create dictionary of geological layer dataframes 
    df_dict = {}

    for sheet_name in xlsx_file.sheet_names:
        df_dict[sheet_name] = xlsx_file.parse(sheet_name)
        
    
    # Drop rows with no or incomplete GSD's
    for key, value in df_dict.items():
        df_dict[key] = value.dropna(subset=["z_90"])

    # Set hole_id as index
    for key, value in df_dict.items():
        df_dict[key] = value.set_index("hole_id")
        
    # Drop unnecessary columns
    df_dict_GSD = {}

    for key, value in df_dict.items():
        df_dict_GSD[key] = value.loc[:, column_range]
        # Check that column order in df matches order of input column_range
        assert (column_range == df_dict_GSD[key].columns).all
    
    # Replace zero by very small value
    for key, value in df_dict_GSD.items():
        df_dict_GSD[key] = value.replace(0, 0.000001)
    
    # Correct percentages so they sum up to exactly 100
    for key, value in df_dict_GSD.items():
        df_dict_GSD[key] = value.divide(value.sum(axis=1), axis=0) * 100
        print(value.shape)
    
    
    # ------------------
    # CLR TRANSFORMATION
    # ------------------
    
    df_dict_GSD_ln = {}

    for key, value in df_dict_GSD.items():
        df_dict_GSD_ln[key] = np.log(value)
        
    df_dict_GSD_clr = {}

    for key, value in df_dict_GSD_ln.items():
        df_dict_GSD_clr[key] = value.subtract(value.mean(axis=1), axis=0)
    
    
    # -----------------------------------
    # PRINCINPAL COMPONENT ANALYSIS (PCA)
    # -----------------------------------
    
    dict_pca = {}
    df_dict_variance = {}

    for key, value in df_dict_GSD_clr.items():

        # Reassign value to overcome copy issues
        data = value
        # Number of components is the minimum of number of rows 
        # and number of columns
        n_comp = min(data.shape)
        pca = PCA(n_components = n_comp)
        pca.fit(data)
        #print(pca.fit(data))

        # Determine PCA expleined variance ratio
        dict_pca[key] = pca
        df_dict_variance[key] = pca.explained_variance_ratio_

        #print(pca.explained_variance_ratio_)

        variance_sum = 0
        variance_n_comp = 0
        #print(data.shape[1])
        for v in range(n_comp):
            #print(pca.explained_variance_ratio_[v])
            if variance_sum < 0.95:
                variance_sum += pca.explained_variance_ratio_[v]
                variance_n_comp += 1
                
        print(variance_n_comp, "PCA components with variance sum", variance_sum, 
              "needed for obtaining sum of variance > 0.95")
    
    df_dict_pca = {}

    for key, value in dict_pca.items():
        #print(key)
        #print(value)
        #print(df_dict_GSD_clr[key])
        #print(value.transform(df_dict_GSD_clr[key]))
        df_dict_pca[key] = pd.DataFrame(value.transform(df_dict_GSD_clr[key]), 
                                        # Change range to (1, max # params + 1)
                                        columns=["PC" + "{:02}".format(i)
                                        for i in range(1,value.n_components+1)], 
                                        index=df_dict_GSD_clr[key].index)
    
    df_dict_pca_merge = {}

    #print(df_dict)

    for key, value in df_dict_pca.items():
        # Does matter from which file the coordinates are taken since 
        # this differs from layer to layer
        df_dict_pca_merge[key] = pd.merge(df_dict[key][["lat", "lon"]], 
                                          value, 
                                          left_index=True, 
                                          right_index=True)
    
    df_variance = pd.DataFrame.from_dict(df_dict_variance).T
    df_variance.columns = ["PC" + "{:02}".format(i) \
                          for i in range(1, df_variance.shape[1] + 1)]
    
    
    # ---------
    # SAVE DATA
    # ---------
    #print(df_dict_pca_merge)
    
    if save_data == True:
        if save_name != None:
            # TO DO: Change default directory to change with input
            writer = pd.ExcelWriter(save_name + ".xlsx")
        else:
            writer = pd.ExcelWriter("../_RESULTS/PCA/Berg_pca.xlsx")

        for key, value in df_dict_pca_merge.items():
            value.to_excel(writer, key)
            print(value.shape)

        df_variance.to_excel(writer, "pca_variance")
        writer.close()

    return [df_dict_GSD_clr, dict_pca, df_dict_pca_merge, df_variance]


def post_processing(directory, df_dict_GSD_clr, dict_pca, grid_data, 
                    n_components=5, save_data=False):
    """Post-processing module of new approach for interpolation 
    of compositional data by ordinary kriging

    Parameters:

    Returns:

    """
    
    # ---------
    # LOAD DATA
    # ---------
    
    # Get quarry code information
    regex0 = re.compile("(data_[A-Za-z]+)")
    quarry_code = regex0.search(directory).group()
    quarry_code = quarry_code[5:]
    
    
    # Get geological layer code information
    regex1 = re.compile("(/[A-Z][A-Z]/)")
    code_geol = regex1.search(directory).group()
    code_geol = code_geol[1:-1]

    try:
        regex2 = re.compile("_\d/")
        n_train = regex2.search(directory).group()
        n_train = n_train[1:-1]
    except Exception as e:
        raise e
    
    print(quarry_code, code_geol)
    
    # Load in PCA component files
    df_dict_pca_kriged = {}
    n_files = 0
    
    # Set up regaular expression to look for PCA component number in filename
    regex2 = re.compile("(PC\d\d)")
    
    # Loop through files
    for file in os.listdir(directory):
        if file.endswith(".xlsx"):
            m = regex2.search(file)
            component = m.group()
            df_dict_pca_kriged[component] = pd.read_excel(directory + "/" 
                                                          + file, header=None)
            print(file, df_dict_pca_kriged[component].shape)
            n_files += 1
    
    panel = pd.Panel(df_dict_pca_kriged)
    
    # Iterate over all values of all dataframes in the panel
    counter = 0
    
    for label, df in panel.iteritems():
        for index, row in df.iterrows():
            for element in row:
                counter += 1
    
    print("Number of grid points per file :", counter/n_files)
    print("Number of grid points in total:", counter)
    
    # Create list of PC scores per kriged gridpoint
    points = []
    
    for i in range(panel.shape[1]):
        for j in range(panel.shape[2]):
            total = []
            for key, value in df_dict_pca_kriged.items():
                total.append(value.loc[i,j])
            points.append(total)
            
    print("gridpoints:", len(points))
    
    # -----------
    # REVERSE PCA
    # -----------

    X = []
    n_comp=n_components
    # Set PCA model properties to use
    # TO DO: Revert to "code_geol"
    model = dict_pca["train"]

    for total in points:
        x = np.dot(total[:n_comp], model.components_[:n_comp, :])
        X.append(x)
    
    
    # --------------------------
    # REVERSE CLR TRANSFORMATION
    # --------------------------

    # Reverse clr - step 1

     # TO DO: Revert to "code_geol"
    clr = df_dict_GSD_clr["train"]
    X_clr = []

    for x in X:
        x += clr.mean(axis=0).values
        X_clr.append(x)
    
    # Reverse clr - step 2
    # Calculate GSD percentages 

    X_ratio = []
    for x_clr in X_clr:
        x_ratio = np.exp(x_clr) / np.sum(np.exp(x_clr)) * 100
        X_ratio.append(x_ratio)
    
    # Check that for every grid point the sum of the percentages equals 100
    for i in range(len(X_ratio)):
        assert np.isclose(pd.DataFrame(X_ratio[i]).sum(), 100.0)
    
    # ---------
    # SAVE DATA
    # ---------
    
    # Group results together per GSD class in grid point matrix

    results = {}
    # TO DO: Check why size of range == 10 (10 grain size classes)
    for i in range(10):
        result = np.empty(df_dict_pca_kriged[component].shape)
        X_index = 0
        #print(X_ratio[0][i])

        for index in np.ndindex(result.shape):

            result[index] = X_ratio[X_index][i]
            X_index += 1
        results[code_geol + str(i)] = result
    
    gsd_classes = ["z_1000", "z_710", "z_500", "z_355", "z_250", 
                   "z_180", "z_125", "z_90", "z_63", "z_0"]
    
    if save_data == True:
        for ((key, value), gsd_class) in zip(results.items(), gsd_classes):
            f = "../_RESULTS/CROSS_VALIDATION_POSTPROCESSED_80-20/" + quarry_code + "/" + code_geol \
                + "/" + str(n_comp) + "comp/" + n_train + "/" + quarry_code + "_" + code_geol \
                + "_" + gsd_class +"_kriged_reverse_" + str(n_comp) + "comp_spherical_" + n_train +".asc"

            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, 'w+') as f:
            #f = open("../_RESULTS/REVERSE_NEW/" + code_geol + "/" + code_geol 
            #         + "_" + gsd_class +"_kriged_reverse_" + str(n_comp) 
            #         + "comp.asc", 'w+')
                f.write(grid_data)
                pd.DataFrame(value).to_csv(f, sep=" ", header=False, 
                                           index=False, mode='a')
    
    return points


def intr_dim(df, n_comp=None, bar=True, cumul=True):
    """Plot variance ratio of PCA to determine intrinsic dimension
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to use for PCA
    n_comp : int (optional)
        Number of components to use in PCA (defaults to None)
    bar : Bool (optional)
        Whether to generate barplot (defaults to True)
    cumul : Bool (optional)
        Whether to generate cumulative plot (defaults to True)
        
    Returns:
    --------
    None
    
    """
    
    # Create pca model
    model_check = PCA(n_components=n_comp)
    
    # FIt pca model
    model_check.fit(df)
    features = range(model_check.n_components_)
    
    # Plot variance ratio per principal component
    fig, ax1 = plt.subplots(nrows=1, sharex=True, sharey=True, figsize=(8,4))
    
    # Barplot
    if bar :
        rects = ax1.bar(features, model_check.explained_variance_ratio_)
        ax1.set_ylabel("Variance ratio")
    
    # Cumulative plot
    if cumul :
        y = np.insert(model_check.explained_variance_ratio_.cumsum(), 0, 0)
        ax1.plot(range(0,10), y[1:], linestyle='none', marker='o')
        ax1.set_ylabel("Cumulative variance ratio")
    

    # Title and axes labels
    #plt.suptitle("Variance ratio per principal component")
    plt.xlabel("Dimensions")
    
    def autolabel(rects):
        """Attach a text label above each bar displaying its height
        
        Parameters:
        -----------
        rects : list
            Bars of barplot
            
        Returns:
        --------
        None
        
        """
        
        for rect in rects:
            i = 0
            height = []
            height.append(rect.get_height())
            ax1.text(rect.get_x() + rect.get_width()/2., height[i],
                     '%2.1f' % (float(height[i])*100),
                      ha='center', va='bottom')
            i += 1
        
        return
    
    
    # Labels
    if bar :
        autolabel(rects)
    
    # Hide major tick labels
    ax1.set_xticklabels('')

    # Customize minor tick labels
    ax1.set_xticks(np.arange(0, 10), minor=True)
    ax1.set_xticklabels(['1','2','3','4','5','6','7','8','9','10'], minor=True)
    
    ax1.set_xlim([-0.5, 9.5])
    #ax1.set_ylim([-0.05, 1.05])
    
    # Margins
    #plt.margins(0.02)
    
    plt.hlines(0.95, -0.5, 9.5)
    plt.text(-0.2, 0.965, "0.95 threshold")
    
    plt.show()
    
    return



#TEST CASE FUNCTIONS

def pre_processing_debug(file, save_data=False, save_name=None):
    """Pre-processing module of new approach for interpolation 
    of compositional data by ordinary kriging

    Parameters:

    Returns:

    """
    
    # -------------------------
    # DATA LOADING AND CLEANING
    # -------------------------
    
    xlsx_file = pd.ExcelFile(file)
    print(xlsx_file.sheet_names)
    
    # Create dictionary of geological layer dataframes 
    df_dict = {}

    for sheet_name in xlsx_file.sheet_names:
        df_dict[sheet_name] = xlsx_file.parse(sheet_name)
        
    
    # Drop rows with no or incomplete GSD's
    # for key, value in df_dict.items():
    #     df_dict[key] = value.dropna(subset=["z_90"])

    # Set hole_id as index
    for key, value in df_dict.items():
        df_dict[key] = value.set_index("hole_id")
        
    # Drop unnecessary columns
    df_dict_GSD = {}

    for key, value in df_dict.items():
        # TO DO: Change column range
        df_dict_GSD[key] = value.loc[:, df_dict[key].columns[2:]]
    
    # Replace zero by very small value
    for key, value in df_dict_GSD.items():
        df_dict_GSD[key] = value.replace(0, 0.000001)
    
    # Correct percentages so they sum up to exactly 100
    for key, value in df_dict_GSD.items():
        df_dict_GSD[key] = value.divide(value.sum(axis=1), axis=0) * 100
    
    
    # ------------------
    # CLR TRANSFORMATION
    # ------------------
    
    df_dict_GSD_ln = {}

    for key, value in df_dict_GSD.items():
        df_dict_GSD_ln[key] = np.log(value)
        
    df_dict_GSD_clr = {}

    for key, value in df_dict_GSD_ln.items():
        df_dict_GSD_clr[key] = value.subtract(value.mean(axis=1), axis=0)
    
    
    # -----------------------------------
    # PRINCINPAL COMPONENT ANALYSIS (PCA)
    # -----------------------------------
    
    dict_pca = {}
    df_dict_variance = {}

    for key, value in df_dict_GSD_clr.items():

        # Reassign value to overcome copy issues
        data = value
        # Number of components is the minimum of number of rows 
        # and number of columns
        n_comp = min(data.shape)
        pca = PCA(n_components = n_comp)
        pca.fit(data)
        print(pca.fit(data))

        # Determine PCA expleined variance ratio
        dict_pca[key] = pca
        df_dict_variance[key] = pca.explained_variance_ratio_

        print(pca.explained_variance_ratio_)

        variance_sum = 0
        variance_n_comp = 0
        print(data.shape[1])
        for v in range(n_comp):
            print(pca.explained_variance_ratio_[v])
            if variance_sum < 0.95:
                variance_sum += pca.explained_variance_ratio_[v]
                variance_n_comp += 1
                
        print(variance_n_comp, "PCA components with variance sum", variance_sum, 
              "needed for obtaining sum of variance > 0.95")
    
    df_dict_pca = {}

    for key, value in dict_pca.items():
        print(key)
        print(value)
        print(df_dict_GSD_clr[key])
        print(value.transform(df_dict_GSD_clr[key]))
        df_dict_pca[key] = pd.DataFrame(value.transform(df_dict_GSD_clr[key]), 
                                        # Change range to (1, max # params + 1)
                                        columns=["PC" + "{:02}".format(i)
                                        for i in range(1,value.n_components+1)], 
                                        index=df_dict_GSD_clr[key].index)
    
    df_dict_pca_merge = {}

    print(df_dict)

    for key, value in df_dict_pca.items():
        # TO DO: Check this line 'df_dict["GZ/Sheet1"]'
        df_dict_pca_merge[key] = pd.merge(df_dict["Sheet2"][["lat", "lon"]], 
                                          value, 
                                          left_index=True, 
                                          right_index=True)
    
    df_variance = pd.DataFrame.from_dict(df_dict_variance).T
    df_variance.columns = ["PC" + "{:02}".format(i) \
                          for i in range(1, df_variance.shape[1] + 1)]
    
    
    # ---------
    # SAVE DATA
    # ---------
    print(df_dict_pca_merge)
    
    if save_data == True:
        if save_name != None:
            writer = pd.ExcelWriter("../_RESULTS/" + save_name + ".xlsx")
        else:
            writer = pd.ExcelWriter("../_RESULTS/Berg_pca.xlsx")

        for key, value in df_dict_pca_merge.items():
            value.to_excel(writer, key)

        df_variance.to_excel(writer, "pca_variance")
        writer.close()

    return [df_dict_GSD_clr, dict_pca, df_dict_pca_merge, df_variance]


def post_processing_debug(directory, df_dict_GSD_clr, dict_pca, grid_data, 
                          n_components=5, save_data=False):
    """Post-processing module of new approach for interpolation 
    of compositional data by ordinary kriging

    Parameters:

    Returns:

    """
    
    # ---------
    # LOAD DATA
    # ---------
    
    # Get quarry code information
    regex0 = re.compile("(data_[A-Za-z]+)")
    quarry_code = regex0.search(directory).group()
    quarry_code = quarry_code[5:]
    
    
    # Get geological layer code information
    regex1 = re.compile("(/[A-Z][A-Z]/)")
    code_geol = regex1.search(directory).group()
    code_geol = code_geol[1:-1]
    
    print(quarry_code, code_geol)
    
    # Load in PCA component files
    df_dict_pca_kriged = {}
    n_files = 0
    
    # Set up regaular expression to look for PCA component number in filename
    regex2 = re.compile("(PC\d\d)")
    
    # Loop through files
    for file in os.listdir(directory):
        if file.endswith(".xlsx"):
            m = regex2.search(file)
            component = m.group()
            df_dict_pca_kriged[component] = pd.read_excel(directory + "/" 
                                                          + file, header=None)
            print(file, df_dict_pca_kriged[component].shape)
            n_files += 1
    
    panel = pd.Panel(df_dict_pca_kriged)
    
    # Iterate over all values of all dataframes in the panel
    counter = 0
    
    for label, df in panel.iteritems():
        for index, row in df.iterrows():
            for element in row:
                counter += 1
    
    print("Number of grid points per file :", counter/n_files)
    print("Number of grid points in total:", counter)
    
    # Create list of PC scores per kriged gridpoint
    points = []
    
    print(panel.shape)

    for i in range(panel.shape[1]):
        for j in range(panel.shape[2]):
            total = []
            for key, value in df_dict_pca_kriged.items():
                total.append(value.loc[i,j])
            points.append(total)
            
    print("gridpoints:", len(points))
    print("points", points[0])
    
    # -----------
    # REVERSE PCA
    # -----------

    X = []
    n_comp=n_components
    # Set PCA model properties to use
    model = dict_pca["Sheet2"] # TO DO: revert to code_geol

    for total in points:
        # Dot product of PC scores and transposed Eigenvectors
        # Number of score and Eigenvectors that is used is determined
        # by the number of components to use for the reconstruction.
        x = np.dot(total[:n_comp], model.components_[:n_comp, :])
        X.append(x)
    
    
    # --------------------------
    # REVERSE CLR TRANSFORMATION
    # --------------------------

    # Reverse clr - step 1

    clr = df_dict_GSD_clr["Sheet2"] # TO DO: revert to code_geol
    X_clr = []

    for x in X:
        x += clr.mean(axis=0).values
        X_clr.append(x)


    
    # Reverse clr - step 2
    # Calculate GSD percentages 

    X_ratio = []
    for x_clr in X_clr:
        x_ratio = np.exp(x_clr) / np.sum(np.exp(x_clr)) * 100
        X_ratio.append(x_ratio)
    
    # Check that for every grid point the sum of the percentages equals 100
    for i in range(len(X_ratio)):
        assert np.isclose(pd.DataFrame(X_ratio[i]).sum(), 100.0)
    
    # ---------
    # SAVE DATA
    # ---------
    
    # Group results together per GSD class in grid point matrix

    results = {}
    # TO DO: Check why size of range == 10 (10 grain size classes)
    for i in range(10): # TO DO: revert to 10
        result = np.empty(df_dict_pca_kriged[component].shape)
        X_index = 0
        #print(X_ratio[0][i])

        for index in np.ndindex(result.shape):

            result[index] = X_ratio[X_index][i]
            X_index += 1
        results[code_geol + str(i)] = result
    
    gsd_classes = ["z_1000", "z_710", "z_500", "z_355", "z_250", 
                   "z_180", "z_125", "z_90", "z_63", "z_0"]
    
    if save_data == True:
        for ((key, value), gsd_class) in zip(results.items(), gsd_classes):
            f = "../_RESULTS/REVERSE_NEW/" + quarry_code + "/" + code_geol \
                + "/" + str(n_comp) + "comp/" + quarry_code + "_" + code_geol \
                + "_" + gsd_class +"_kriged_reverse_" + str(n_comp) + "comp.asc"

            os.makedirs(os.path.dirname(f), exist_ok=True)
            with open(f, 'w+') as f:
            #f = open("../_RESULTS/REVERSE_NEW/" + code_geol + "/" + code_geol 
            #         + "_" + gsd_class +"_kriged_reverse_" + str(n_comp) 
            #         + "comp.asc", 'w+')
                f.write(grid_data)
                pd.DataFrame(value).to_csv(f, sep=" ", header=False, 
                                           index=False, mode='a')
    
    return points