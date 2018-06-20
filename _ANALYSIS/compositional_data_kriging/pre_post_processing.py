import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
# will be needed when panels in pandas gets deprecated
# import xarray

# ===================================================================
# TO DO:
#  - Replace soon to be deprecated panels of pandas by xarray package
# ===================================================================


def pre_processing(file, save_data=False, save_name=None,
                   column_range=["z_1000", "z_710", "z_500", "z_355", "z_250",
                                 "z_180", "z_125", "z_90", "z_63", "z_0"]):
    """Pre-processing module of new approach for interpolation
    of compositional data by ordinary kriging

    Parameters:
    -----------
    file : str
        Path to file to be pre-processed
    save_data : Bool (optional)
        Whether to save pre-processed data or not (defaults to False)
    save_name : str (optional)
        If save_data==True, path to save pre-processed files to
        (defaults to None)
    column_range : list (optional)
        List of column headers to use as lookup columns for
        compositional data in input file
        (defaults to topic-related grain size class names)

    Returns:
    --------
    List of
        df_dict_GSD_clr :  dict
            Dictionary of dataframes with centred log ratio (clr)
            transformd grain size data
        dict_pca : dict
            Dictionary of PCA model parameters
        df_dict_pca_merge : dict
            Dictionary of dataframes with principal component scores
        df_variance
            Dataframe with PCA explained variance

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
        pca = PCA(n_components=n_comp)
        pca.fit(data)
        # print(pca.fit(data))

        # Determine PCA expleined variance ratio
        dict_pca[key] = pca
        df_dict_variance[key] = pca.explained_variance_ratio_

        # print(pca.explained_variance_ratio_)

        variance_sum = 0
        variance_n_comp = 0
        # print(data.shape[1])
        for v in range(n_comp):
            # print(pca.explained_variance_ratio_[v])
            if variance_sum < 0.95:
                variance_sum += pca.explained_variance_ratio_[v]
                variance_n_comp += 1

        print(variance_n_comp, "PCA components with variance sum",
              variance_sum, "needed for obtaining sum of variance > 0.95")

    df_dict_pca = {}

    for key, value in dict_pca.items():
        # print(key)
        # print(value)
        # print(df_dict_GSD_clr[key])
        # print(value.transform(df_dict_GSD_clr[key]))
        df_dict_pca[key] = pd.DataFrame(value.transform(df_dict_GSD_clr[key]),
                                        # Change range to (1, max # params + 1)
                                        columns=["PC" + "{:02}".format(i)
                                                 for i in
                                                 range(1, value.n_components
                                                       + 1)],
                                        index=df_dict_GSD_clr[key].index)

    df_dict_pca_merge = {}

    # print(df_dict)

    for key, value in df_dict_pca.items():
        # Does matter from which file the coordinates are taken since
        # this differs from layer to layer
        df_dict_pca_merge[key] = pd.merge(df_dict[key][["lat", "lon"]],
                                          value,
                                          left_index=True,
                                          right_index=True)

    df_variance = pd.DataFrame.from_dict(df_dict_variance).T
    df_variance.columns = ["PC" + "{:02}".format(i)
                           for i in range(1, df_variance.shape[1] + 1)]

    # ---------
    # SAVE DATA
    # ---------
    # print(df_dict_pca_merge)

    if save_data is True:
        if save_name is not None:
            # TO DO: Change default directory to change with input
            writer = pd.ExcelWriter(save_name + ".xlsx")
        else:
            writer = pd.ExcelWriter("../_RESULTS/PREPROCESSED/pca.xlsx")

        for key, value in df_dict_pca_merge.items():
            value.to_excel(writer, key)
            print(value.shape)

        df_variance.to_excel(writer, "pca_variance")
        writer.close()

    return [df_dict_GSD_clr, dict_pca, df_dict_pca_merge, df_variance]


def post_processing(directory, df_dict_GSD_clr, dict_pca, grid_info,
                    n_components=5, save_data=False, input_format="xlsx",
                    verify=True):
    """Post-processing module of new approach for interpolation
    of compositional data by ordinary kriging

    Parameters:
    -----------
    directory : str
        Path of directory
    df_dict_GSD_clr : dict
        Dictionary of dataframes with centred log ratio (clr)
        transformd grain size data (resulting from preprocessing)
    dict_pca : dict
        Dictionary of PCA model parameters
        (resulting from pre-processing)
    grid_info : str
        Grid file parameters
    n_components : int (optional)
        Number of principal components to use during reverse PCA
        (defaults to 5)
    save_data : Bool (optional)
        Whether to save the data or not (defaults to False)
    input_format : str (optional)
        Data format to look for in directories to use as input files
        defaults to .xlsx
    verify : Bool (optional)
        Whether to verify the unit-sum constraint on the resulting
        grain size data as percentages (defaults to True)


    Returns:
    --------
    points : list
        List of individual grid file points

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
        pass

    print(quarry_code, code_geol)

    # Load in PCA component files
    df_dict_pca_kriged = {}
    n_files = 0

    # Set up regaular expression to look for PCA component number in filename
    regex2 = re.compile("(PC\d\d)")

    # Loop through files
    for file in os.listdir(directory):
        if input_format == "xlsx":
            if file.endswith(".xlsx"):
                m = regex2.search(file)
                component = m.group()
                df_dict_pca_kriged[component] = pd.read_excel(directory + "/"
                                                              + file,
                                                              header=None)
                print(file, df_dict_pca_kriged[component].shape)
                n_files += 1
        elif input_format == "csv":
            if file.endswith(".csv"):
                m = regex2.search(file)
                component = m.group()
                df_dict_pca_kriged[component] = pd.read_csv(directory + "/"
                                                            + file, sep=";",
                                                            header=None)
                print(file, df_dict_pca_kriged[component].shape)
                n_files += 1
        else:
            "Error"

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
                total.append(value.loc[i, j])
            points.append(total)

    print("gridpoints:", len(points))

    # -----------
    # REVERSE PCA
    # -----------

    X = []
    n_comp = n_components
    # Set PCA model properties to use
    try:
        model = dict_pca[code_geol]
    except KeyError:
        model = dict_pca["train"]

    for total in points:
        x = np.dot(total[:n_comp], model.components_[:n_comp, :])
        X.append(x)

    # --------------------------
    # REVERSE CLR TRANSFORMATION
    # --------------------------

    # Reverse clr - step 1

    try:
        clr = df_dict_GSD_clr[code_geol]
    except KeyError:
        clr = df_dict_GSD_clr["train"]

    X_clr = []

    for x in X:
        x += clr.mean(axis=0).values
        X_clr.append(x)
    # Not faster
    # X_clr = map(lambda x: x + clr.mean(axis=0).values, X)

    # Reverse clr - step 2
    # Calculate GSD percentages

    X_ratio = []
    for x_clr in X_clr:
        x_ratio = np.exp(x_clr) / np.sum(np.exp(x_clr)) * 100
        X_ratio.append(x_ratio)

    # Check that for every grid point the sum of the percentages equals 100
    if verify is True:
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
        # print(X_ratio[0][i])

        for index in np.ndindex(result.shape):

            result[index] = X_ratio[X_index][i]
            X_index += 1
        results[code_geol + str(i)] = result

    gsd_classes = ["z_1000", "z_710", "z_500", "z_355", "z_250",
                   "z_180", "z_125", "z_90", "z_63", "z_0"]

    if save_data is True:
        for ((key, value), gsd_class) in zip(results.items(), gsd_classes):
            f = f"../_RESULTS/CROSS_VALIDATION_POSTPROCESSED_80-20/\
                 {quarry_code}/{code_geol}/{str(n_comp)}comp/{n_train}/\
                 {quarry_code}_{code_geol}_{gsd_class}_kriged_reverse_\
                 {str(n_comp)}comp_spherical_{n_train}.asc"

            os.makedirs(os.path.dirname(f), exist_ok=True)
            # f = open("../_RESULTS/REVERSE_NEW/" + code_geol + "/" + code_geol
            #         + "_" + gsd_class +"_kriged_reverse_" + str(n_comp)
            #         + "comp.asc", 'w+')
            with open(f, 'w+') as f:

                f.write(grid_info)
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
    fig, ax1 = plt.subplots(nrows=1, sharex=True, sharey=True, figsize=(8, 4))

    # Barplot
    if bar:
        rects = ax1.bar(features, model_check.explained_variance_ratio_)
        ax1.set_ylabel("Variance ratio")

    # Cumulative plot
    if cumul:
        y = np.insert(model_check.explained_variance_ratio_.cumsum(), 0, 0)
        ax1.plot(range(0, 10), y[1:], linestyle='none', marker='o')
        ax1.set_ylabel("Cumulative variance ratio")

    # Title and axes labels
    # plt.suptitle("Variance ratio per principal component")
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
    if bar:
        autolabel(rects)

    # Hide major tick labels
    ax1.set_xticklabels('')

    # Customize minor tick labels
    ax1.set_xticks(np.arange(0, 10), minor=True)
    ax1.set_xticklabels(['1', '2', '3', '4', '5', '6',
                        '7', '8', '9', '10'], minor=True)

    ax1.set_xlim([-0.5, 9.5])
    # ax1.set_ylim([-0.05, 1.05])

    # Margins
    # plt.margins(0.02)

    plt.hlines(0.95, -0.5, 9.5)
    plt.text(-0.2, 0.965, "0.95 threshold")

    plt.show()

    return


def pre_pre_processing(file, column_range=["z_1000", "z_710", "z_500", "z_355",
                                           "z_250", "z_180", "z_125", "z_90",
                                           "z_63", "z_0"]):
    """Perform clr-transformation before actual pre-processing

    Parameters:
    -----------
    file : str
        Path to file to be pre-processed
    column_range : list (optional)
        List of column headers to use as lookup columns for
        compositional data in input file
        (defaults to topic-related grain size class names)

    Returns:
    --------
    df_dict_clr_merge : dict
        Dictionary of dataframes with clr transformed data
        merged with coordinates

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

    df_dict_clr_merge = {}

    # print(df_dict)

    for key, value in df_dict_GSD_clr.items():
        # Does matter from which file the coordinates are taken since
        # this differs from layer to layer
        df_dict_clr_merge[key] = pd.merge(df_dict[key][["lat", "lon"]],
                                          value,
                                          left_index=True,
                                          right_index=True)

    return df_dict_clr_merge


def pca_pre_processing(file, save_data=False, save_name=None,
                       column_range=["z_1000", "z_710", "z_500", "z_355",
                                     "z_250", "z_180", "z_125", "z_90", "z_63",
                                     "z_0"]):
    """Perform PCA on already clr-transformed data"""

    xlsx_file = pd.ExcelFile(file)
    print(xlsx_file.sheet_names)

    # Create dictionary of geological layer dataframes
    df_dict = {}

    for sheet_name in xlsx_file.sheet_names:
        df_dict[sheet_name] = xlsx_file.parse(sheet_name, index_col='hole_id')

    # Drop unnecessary columns
    df_dict_GSD_clr = {}

    for key, value in df_dict.items():
        df_dict_GSD_clr[key] = value.loc[:, column_range]

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
        pca = PCA(n_components=n_comp)
        pca.fit(data)
        # print(pca.fit(data))

        # Determine PCA expleined variance ratio
        dict_pca[key] = pca
        df_dict_variance[key] = pca.explained_variance_ratio_

        # print(pca.explained_variance_ratio_)

        variance_sum = 0
        variance_n_comp = 0
        # print(data.shape[1])
        for v in range(n_comp):
            # print(pca.explained_variance_ratio_[v])
            if variance_sum < 0.95:
                variance_sum += pca.explained_variance_ratio_[v]
                variance_n_comp += 1

        print(variance_n_comp, "PCA components with variance sum",
              variance_sum, "needed for obtaining sum of variance > 0.95")

    df_dict_pca = {}

    for key, value in dict_pca.items():
        # print(key)
        # print(value)
        # print(df_dict_GSD_clr[key])
        # print(value.transform(df_dict_GSD_clr[key]))
        df_dict_pca[key] = pd.DataFrame(value.transform(df_dict_GSD_clr[key]),
                                        # Change range to (1, max # params + 1)
                                        columns=["PC" + "{:02}".format(i)
                                                 for i in
                                                 range(1, value.n_components
                                                       + 1)],
                                        index=df_dict_GSD_clr[key].index)

    df_dict_pca_merge = {}

    # print(df_dict)

    for key, value in df_dict_pca.items():
        # TO DO: Check this line 'df_dict["GZ/Sheet1"]'
        # Does matter from which file the coordiantes are taken since
        # this differs from layer to layer
        df_dict_pca_merge[key] = pd.merge(df_dict[key][["lat", "lon"]],
                                          value,
                                          left_index=True,
                                          right_index=True)
        print(df_dict[key].index)

    df_variance = pd.DataFrame.from_dict(df_dict_variance).T
    df_variance.columns = ["PC" + "{:02}".format(i)
                           for i in range(1, df_variance.shape[1] + 1)]

    # ---------
    # SAVE DATA
    # ---------
    # print(df_dict_pca_merge)

    if save_data is True:
        if save_name is not None:
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


def pca_post_processing(directory, df_dict_GSD_clr, dict_pca, grid_info,
                        n_components=5, save_data=False, input_format="xlsx",
                        verify=True):
    """Post-processing module of new approach for interpolation
    of compositional data by ordinary kriging

    Parameters:
    -----------
    directory : str
        Path of directory
    df_dict_GSD_clr : dict
        Dictionary of dataframes with centred log ratio (clr)
        transformd grain size data (resulting from preprocessing)
    dict_pca : dict
        Dictionary of PCA model parameters
        (resulting from pre-processing)
    grid_info : str
        Grid file parameters
    n_components : int (optional)
        Number of principal components to use during reverse PCA
        (defaults to 5)
    save_data : Bool (optional)
        Whether to save the data or not (defaults to False)
    input_format : str (optional)
        Data format to look for in directories to use as input files
        defaults to .xlsx
    verify : Bool (optional)
        Whether to verify the unit-sum constraint on the resulting
        grain size data as percentages (defaults to True)

    Returns:
    --------
    points : list
        List of individual grid file points

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
        if input_format == "xlsx":
            if file.endswith(".xlsx"):
                m = regex2.search(file)
                component = m.group()
                df_dict_pca_kriged[component] = pd.read_excel(directory + "/"
                                                              + file,
                                                              header=None)
                print(file, df_dict_pca_kriged[component].shape)
                n_files += 1
        elif input_format == "csv":
            if file.endswith(".csv"):
                m = regex2.search(file)
                component = m.group()
                df_dict_pca_kriged[component] = pd.read_csv(directory + "/"
                                                            + file, sep=";",
                                                            header=None)
                print(file, df_dict_pca_kriged[component].shape)
                n_files += 1
        else:
            "Error"

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
                total.append(value.loc[i, j])
            points.append(total)

    print("gridpoints:", len(points))

    # -----------
    # REVERSE PCA
    # -----------

    X = []
    n_comp = n_components
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

    X_ratio = X_clr

    # Reverse clr - step 2
    # Calculate GSD percentages

#     X_ratio = []
#     for x_clr in X_clr:
#         x_ratio = np.exp(x_clr) / np.sum(np.exp(x_clr)) * 100
#         X_ratio.append(x_ratio)

    # Check that for every grid point the sum of the percentages equals 100
    if verify is True:
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
        # print(X_ratio[0][i])

        for index in np.ndindex(result.shape):

            result[index] = X_ratio[X_index][i]
            X_index += 1
        results[code_geol + str(i)] = result

    gsd_classes = ["z_1000", "z_710", "z_500", "z_355", "z_250",
                   "z_180", "z_125", "z_90", "z_63", "z_0"]

    if save_data is True:
        for ((key, value), gsd_class) in zip(results.items(), gsd_classes):
            f = f"../_RESULTS/CROSS_VALIDATION_POSTPROCESSED_80-20/\
                 {quarry_code}/{code_geol}/{str(n_comp)}comp/{n_train}/\
                 {quarry_code}_{code_geol}_{gsd_class}_kriged_reverse_\
                 {str(n_comp)}comp_spherical_{n_train}.asc"

            os.makedirs(os.path.dirname(f), exist_ok=True)
            # f = open("../_RESULTS/REVERSE_NEW/" + code_geol + "/"
            #          + code_geol + "_" + gsd_class +"_kriged_reverse_"
            #          + str(n_comp) + "comp.asc", 'w+')
            with open(f, 'w+') as f:

                f.write(grid_info)
                pd.DataFrame(value).to_csv(f, sep=" ", header=False,
                                           index=False, mode='a')

    return points


def manipulate_non_negativity(data):
    """Apply non-negativity constraint by manipulating the data.
    This is done by putting all negative values to zero

    Parameters:
    -----------
    data : pd.DataFrame
        Original data

    Returns:
    --------
    data_noneg : pd.DataFrame
        Manipulated data to which non-negativity
        constraint has been applied
    """

    # Create copy of data to overcome mirroring issues
    data_copy = data.copy()
    # Apply non-negativity filter and change to zero
    data_copy[data_copy < 0.0] = 0.0
    data_noneg = data_copy

    return data_noneg


def manipulate_constant_sum(data):
    """Apply constant sum constraint by manipulating the data.
    This is done by normalizing the data again

    Parameters:
    -----------
    data : pd.DataFrame
        Original data

    Returns:
    --------
    data_cstsum : pd.DataFrame
        Manipulated data to which constant sum
        constraint has been applied
    """

    # Create copy of data to overcome mirroring issues
    data_copy = data.copy()
    # Apply constant sum constraint by renormalizing
    data_cstsum = data_copy.divide(data_copy.sum(axis=1), axis=0)

    return data_cstsum
