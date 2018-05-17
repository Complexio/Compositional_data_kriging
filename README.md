# Compositional_data_kriging
M.Sc. thesis topic of Matthias Van Delsen Optimal geostatistical interpolation of complete grain-size distributions for estimation of sand resources

## Idea
Finding a better way to interpolate compositional data (here in the form of grain size data) by using centred log ratio (clr) transformation and PCA on the data, using ordinary kriging to interpolate them and finally reversing the PCA and clr transformation to obtain compositional data.

## Folder structure:
|├───_ANALYSIS
│   ├───.ipynb_checkpoints
│   └───borehole_analysis
│       └───__pycache__
├───_DATA
│   ├───Berg_groeve
│   └───MHZ_groeve
├───_FIGURES
├───_KRIGING
│   ├───Kriged_pca_data_Berg
│   │   ├───GZ
│   │   ├───IZ
│   │   └───TZ
│   └───Kriged_pca_data_MHZ
│       ├───GZ
│       ├───IZ
│       └───TZ
├───_RESULTS
│   ├───PCA
│   ├───PREPROCESSED
│   ├───REVERSE
│   │   ├───GZ
│   │   │   ├───5comp
│   │   │   └───9comp
│   │   ├───IZ
│   │   │   ├───5comp
│   │   │   └───9comp
│   │   ├───PCA
│   │   └───TZ
│   │       ├───5comp
│   │       └───9comp
│   ├───REVERSE_NEW
│   │   ├───Berg
│   │   │   ├───GZ
│   │   │   │   ├───1comp
│   │   │   │   ├───2comp
│   │   │   │   ├───3comp
│   │   │   │   ├───4comp
│   │   │   │   ├───5comp
│   │   │   │   ├───6comp
│   │   │   │   ├───7comp
│   │   │   │   ├───8comp
│   │   │   │   └───9comp
│   │   │   ├───IZ
│   │   │   │   ├───1comp
│   │   │   │   ├───2comp
│   │   │   │   ├───3comp
│   │   │   │   ├───4comp
│   │   │   │   ├───5comp
│   │   │   │   ├───6comp
│   │   │   │   ├───7comp
│   │   │   │   ├───8comp
│   │   │   │   └───9comp
│   │   │   └───TZ
│   │   │       ├───1comp
│   │   │       ├───2comp
│   │   │       ├───3comp
│   │   │       ├───4comp
│   │   │       ├───5comp
│   │   │       ├───6comp
│   │   │       ├───7comp
│   │   │       ├───8comp
│   │   │       └───9comp
│   │   └───MHZ
│   │       ├───GZ
│   │       │   ├───1comp
│   │       │   ├───2comp
│   │       │   ├───3comp
│   │       │   ├───4comp
│   │       │   ├───5comp
│   │       │   ├───6comp
│   │       │   ├───7comp
│   │       │   ├───8comp
│   │       │   └───9comp
│   │       ├───IZ
│   │       │   ├───1comp
│   │       │   ├───2comp
│   │       │   ├───3comp
│   │       │   ├───4comp
│   │       │   ├───5comp
│   │       │   ├───6comp
│   │       │   ├───7comp
│   │       │   ├───8comp
│   │       │   └───9comp
│   │       └───TZ
│   │           ├───1comp
│   │           ├───2comp
│   │           ├───3comp
│   │           ├───4comp
│   │           ├───5comp
│   │           ├───6comp
│   │           ├───7comp
│   │           ├───8comp
│   │           └───9comp
│   └───SURFER
├───_SCRIPTS
└───_TABLES
