import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def subset_df_based_on_categories(df, category_column):
    """Create subsets of a dataframe based on a category column

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to be subsetted
    category_column : str
        Column name which contains the categories
        to be used for subsetting the dataframe

    Returns:
    --------
    subsets : dict
        Dictionary of categories : subset dataframes
    """

    categories = df[category_column].unique()
    # df_names = [df.name + "_" + name for name in categories]
    subsets = {}

    for cat in categories:
        subsets[cat] = df[df[category_column] == cat]

    return subsets


def combine_id_rows_simple(df, category, id_column, from_column, to_column):
    """Combine similar id rows by creating ranges from
    minimum and maximum values so that dataframe with
    unique rows is returned

    Parameters:
    -----------
    subset : pd.DataFrame
        Dataframe to combine rows of
    id_column : str
        Name of the id column to use for combining rows
    from_column : str
        Name of the column to use for minimum value
    to_column : str
        Name of the column to use for maximum value

    Returns:
    --------
    df_unique (pd.DataFrame)
        Dataframe with unique rows and new ranges
    """

    ids = df[id_column].unique()
    unique = []

    for hole_id in ids:
        df_id = df[df[id_column] == hole_id]
        minimum = df_id[from_column].min()
        maximum = df_id[to_column].max()
        # CHANGE "IZ" to variable that chooses name based on category
        unique.append([hole_id, minimum, maximum, category])

    df_unique = pd.DataFrame(unique, columns=[id_column,
                                              from_column,
                                              to_column,
                                              "code_geol"])

    # Check that number of unique values of returned df
    # equals number of unique values of initial subsetted df
    assert df_unique.shape[0] == len(df[id_column].unique())

    return df_unique


def combine_id_rows_complex(df, category, id_column, from_column, to_column):
    """Combine similar id rows by creating ranges from
    minimum and maximum values so that dataframe with
    unique rows is returned while also averaging values
    of all subsequent columns

    Parameters:
    -----------
    subset : pd.DataFrame
        Dataframe to combine rows of
    id_column : str
        Name of the id column to use for combining rows
    from_column : str
        Name of the column to use for minimum value
    to_column : str
        Name of the column to use for maximum value

    Returns:
    --------
    df_unique : pd.DataFrame
        Dataframe with unique rows and new ranges
    """

    ids = df[id_column].unique()
    df_averages = pd.DataFrame()
    unique = []

    for hole_id in ids:
        df_id = df[df[id_column] == hole_id]
        minimum = df_id[from_column].min()
        maximum = df_id[to_column].max()

        unique.append([hole_id, minimum, maximum, category])

        # Mind the transitions between layers and do not
        # include them in the averaging?
        df_data_averaged = df_id.groupby("hole_id").mean().reset_index().\
            drop([from_column, to_column], axis=1)
        # print(df_data_averaged)
        df_averages = df_averages.append(df_data_averaged)

    df_unique = pd.DataFrame(unique, columns=[id_column, from_column,
                                              to_column, "code_geol"])
    # print(category, "unique:", df_unique.shape)
    # print(category, "averages:", df_averages.shape)
    try:
        df_unique_averages = df_unique.merge(df_averages,
                                             on=id_column,
                                             how='inner')
        print(category, "merge:", df_unique_averages.shape)
    except ValueError:
        print('error')

    # Check that number of unique values of returned df
    # equals number of unique values of initial subsetted df
    assert df_unique.shape[0] == len(df[id_column].unique())

    return df_unique_averages


def plot_borehole(df, hole_ids):
    """Plot borehole using units

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with all borehole data
    hole_id : list(str)
        Name of the borehole to plot
    Return:
    -------
    None

    """
    # Colors to use per unit in plot
    colordict = {"OVB": "limegreen",
                 "GR": "r",
                 "TZ": "gold",
                 "BC": "dodgerblue",
                 "IZ": "grey",
                 "GZ": "orangered"}

    for hole_id in hole_ids:
        # df["color"] = df["code_geol"].apply(lambda x: colordict[x])
        hole = df[df["hole_id"] == hole_id].copy()
        hole["color"] = hole["code_geol"].apply(lambda x: colordict[x])

        # Iterate over all units of the hole
        fig, ax = plt.subplots()

        for index, row in hole.iterrows():
            interval = hole.loc[index, "depth_from"] \
                            - hole.loc[index, "depth_to"]
            bottom = np.negative(hole.loc[index, "depth_from"])

            plt.bar(np.arange(1), interval, width=0.1,
                    bottom=bottom, color=hole.loc[index, "color"])
            # plt.text(0, (bottom + interval/2), hole.loc[index, "code_geol"])
            plt.title("borehole " + hole_id)
            plt.ylabel("depth (m)")

        labels = [plt.Line2D([0, 0], [0, 0],
                             color=color,
                             marker='s',
                             linestyle="") for color in colordict.values()]

        legend = plt.legend(labels,
                            list(colordict.keys()),
                            numpoints=1,
                            loc="upper right",
                            frameon=True)

        plt.gca().add_artist(legend)

        plt.show()

    return
