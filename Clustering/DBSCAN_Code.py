# Supervised and Unsupervised ML algorithms to understand the restoration potential of Acacia mangium
# DBSCAN analysis

#region Library installation ####
# Unsupervised ML
# Install library for k-means in command prompt (Shell in Windows)
#pip install matplotlib
#pip install kneed
#pip install scikit-learn
#pip install pandas # for data processing
#pip install numpy # for linear algebra
#pip install seaborn # for statistical data visualization
#endregion

#region Python library import ####
import matplotlib.pyplot as plt # for data visualisation
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder # for categorical data to use as labels
from sklearn.decomposition import PCA
from sklearn import metrics
# Specific library for DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from matplotlib.colors import ListedColormap # for visualisation
from sklearn.cluster import AgglomerativeClustering
# Specific for AHC
from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram, fcluster
import scipy.cluster.hierarchy as shc
from sklearn.metrics import confusion_matrix
# Specific for Supervised ML
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier #for KNN
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from scipy.spatial import ConvexHull #for scatter plots
from sklearn.feature_selection import VarianceThreshold # for feature selection
from sklearn import svm # for svm
from sklearn import metrics # for svm model evaluation
import statistics
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import argparse
from sklearn.model_selection import GridSearchCV
#endregion

def set_seed(random_seed=0):
    np.random.seed(random_seed)
    return random_seed

def data_preprocessing(dataset, verbose=False):
    """
    This function is to preprocess the data, removing unsuited samples,
    removing the first column from the dataset of the variables (the first column has the labels of the data),
    and converting into integers the numeric values and scaling the whole dataset.
    @params:
        dataset (dataset, pandas or numpy): dataset to work on
        verbose (bool): specify if to print or not the steps and their result
    @returns: dataset divided into X and Y, where X is the dataset with features and samples, and Y the column with the labels of the samples.
    """
    df = pd.read_csv(dataset)
    # Exclusion of samples I2NB, O2NB, O4NA, R1SA, R1SB because bacteria and fungi data with
    # not enough sequencing depth
    # Use boolean indexing to filter out the rows
    # Specify names to remove
    names_to_remove = ['I2NB', 'O2NB', 'O4NA', 'R1SA', 'R1SB']
    df = df[~df['Sample_ID'].isin(names_to_remove)]
    if verbose:
        print("df.shape: ", df.shape)
        print("df: ", df)
    # Now remove the first column, which is "Sample_ID"
    df = df.iloc[:, 1:]
    if verbose: # Print the filtered DataFrame
        print(df)
        print("df.shape: ", df.shape)
    # Python should already see that all columns have numeric data besides the first two
    # Statistical summary of the dataset
    # print(df.describe())
    # Declare feature vector and target variable
    # Now is treated as numpy df
    X = df.iloc[:,1:] # select all rows and all the columns from the third, excluding the first 2 categorical
    if verbose:
        print("X.shape: ", X.shape)
        print("X: ", X)
    Y = df['Landcover']
    if verbose:
        print("Y.shape: ", Y.shape)
        print("Y: ", Y)
    # Convert categorical variable into integers
    map_landcover_label = {"Grassland": 0, "2 years old": 1, "10 years old": 2, "24 years old": 3, "Remnant": 4}
    if verbose:
        print("Y before mapping: ", Y)
    Y = [map_landcover_label[lab] for lab in Y]
    if verbose:
        print("Y after mapping: ", Y)
    #le = LabelEncoder() # This 2 lines to implement to change into integers the labels without specifications
    #Y = le.fit_transform(Y)
    if verbose:
        print("Y: ", Y)
    # Feature scaling
    ms = StandardScaler()
    X = ms.fit_transform(X)
    X = pd.DataFrame(X)
    if verbose:
        print("X after scaling: ", X)
    return X,Y

def pca_dim_reduction(dataset_original, dataset, n_dim, verbose=False, plots=False):
    """
    This function is to implement feature extraction by reducing the X dataset dimensions through PCA analysis, retaining 2 PCs.
    @params:
        dataset (dataset, pandas or numpy): dataset to work on
        n_dim: number of features to extract
        verbose (bool): specify if to print or not the steps and their result
    @returns: X dataset with dimensions reduced to n_dim
    """
    df = pd.read_csv(dataset_original)
    df = pd.read_csv(dataset_original)
    names_to_remove = ['I2NB', 'O2NB', 'O4NA', 'R1SA', 'R1SB']
    df = df[~df['Sample_ID'].isin(names_to_remove)]
    # Now remove the first column, which is "Sample_ID"
    df = df.iloc[:, 1:]
    # Python should already see that all columns have numeric data besides the first two
    # Statistical summary of the dataset
    # print(df.describe())
    # Declare feature vector and target variable
    # Now is treated as numpy df
    X = df.iloc[:,1:] # select all rows and all the columns from the third, excluding the first 2 categorical
    pca = PCA(n_dim) # Reduce dataset to n_dim dimensions
    XPCA = pca.fit_transform(dataset)
    # Check the variance of the components
    if plots:
        plt.figure(figsize=(10,10))
        var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
        lbls = [str(x) for x in range(1,len(var)+1)]
        plt.bar(x=range(1,len(var)+1), height = var, tick_label = lbls)
        plt.show()
    # Get and display loadings (Original Variable Contributions)
    # The pca.components_ attribute contains the loadings
    # Each row is a PC, each column is an original feature
    loadings = pd.DataFrame(pca.components_,
                            columns=X.columns, # Use the correctly determined feature names
                            index=[f'PC{i+1}' for i in range(pca.n_components_)])
    if verbose:
        print("--- PCA Explained Variance ---")
        # Explained Variance Ratio for PC1
        explained_variance_pc1 = pca.explained_variance_ratio_[0]
        print(f"Explained Variance Ratio by PC1: {explained_variance_pc1:.4f}")
        # Explained Variance Ratio for PC2
        explained_variance_pc2 = pca.explained_variance_ratio_[1]
        print(f"Explained Variance Ratio by PC2: {explained_variance_pc2:.4f}")
        print("\n--- Principal Component Loadings (Original Variable Contributions) ---")
        print("Each row is a PC, each column is an original feature.")
        print("Values indicate the weight/contribution of each original feature to that PC.")
        print("Larger absolute values signify a stronger contribution.")
        print(loadings)
        # Top 10 variables contribution to the first 2 PCs
        print("\n--- Contributions of Original Variables for the First 2 PCs ---")
        if n_dim >= 1:
            print("\nPC1 Loadings (Contributions):")
            pc1_contributions = loadings.loc['PC1'].sort_values(key=abs, ascending=False).head(10)
            print(pc1_contributions) # Sort by absolute value for contribution, False will sort ascending, True will sort descending
        if n_dim >= 2:
            print("\nPC2 Loadings (Contributions):")
            pc2_contributions = loadings.loc['PC2'].sort_values(key=abs, ascending=False).head(10)
            print(pc2_contributions)
    # Make a table of PCs 1 and 2 and save in csv
    # Save the combined results if output_dir is provided
    # Top 10 variables contribution to the first 2 PCs
    all_top_contributions = pd.DataFrame()
    if n_dim >= 1:
        pc1_contributions = loadings.loc['PC1'].sort_values(key=abs, ascending=False).head(10)
        # Convert Series to DataFrame with a descriptive column name
        pc1_df = pc1_contributions.rename('PC1_Loading').to_frame()
        all_top_contributions = pd.concat([all_top_contributions, pc1_df], axis=1)
    if n_dim >= 2:
        pc2_contributions = loadings.loc['PC2'].sort_values(key=abs, ascending=False).head(10)
        # Convert Series to DataFrame with a descriptive column name
        pc2_df = pc2_contributions.rename('PC2_Loading').to_frame()
        # Use join to merge based on index (feature names)
        all_top_contributions = all_top_contributions.join(pc2_df, how='outer')
    # Define the filename for the combined contributions
    all_top_contributions.to_csv(f'DBSCAN_top10_PC1_PC2_combined_contributions_with_seed_{args.random_seed}.csv')
    print(f"\nSaved combined top 10 contributions for PC1 and PC2")
    
    return XPCA

def variance_tresh_dim_reduction(dataset_original,variance_threshold, verbose=False):
    """
    This function is to implement feature selection by reducing the X dataset dimensions through Variance Treshold analysis, 
    retaining n features according to the threshold selected.
    @params:
        dataset_original (dataset, pandas or numpy): original dataset to work on
        variance_threshold: threshold of variance to exclude variables
        verbose (bool): specify if to print or not the steps and their result
    @returns: X dataset with dimensions reduced according to the threshold selected.
    """
    df = pd.read_csv(dataset_original)
    names_to_remove = ['I2NB', 'O2NB', 'O4NA', 'R1SA', 'R1SB']
    df = df[~df['Sample_ID'].isin(names_to_remove)]
    if verbose:
        print("df.shape: ", df.shape)
        print("df: ", df)
    # Now remove the first column, which is "Sample_ID"
    df = df.iloc[:, 1:]
    if verbose: # Print the filtered DataFrame
        print(df)
        print("df.shape: ", df.shape)
    # Python should already see that all columns have numeric data besides the first two
    # Statistical summary of the dataset
    # print(df.describe())
    # Declare feature vector and target variable
    # Now is treated as numpy df
    X = df.iloc[:,1:] # select all rows and all the columns from the third, excluding the first 2 categorical
    if verbose:
        print("X.shape: ", X.shape)
        print("X: ", X)
    # Feature scaling
    ms = MinMaxScaler()
    X_transformed = ms.fit_transform(X)
    X_for_variance = pd.DataFrame(X_transformed, columns=X.columns)
    variance = X_for_variance.var()
    if verbose:
        print("Variance:", variance)
    variance.to_csv("VarianceFeaturesClustering.csv") # Save to Excel file to analyse better and explore the treshold to retain 2, 3, or n number of variables.
    VT = VarianceThreshold(threshold=variance_threshold)
    XVT = VT.fit_transform(X_for_variance)
    selected_features = X_for_variance.columns[VT.get_support()]
    XVT = pd.DataFrame(XVT, columns=selected_features)
    if verbose:
        print("XVT columns:", XVT) 
    return XVT

def plot_k_distance_graph(dataset, k, data_type="raw", verbose=False, see_plots=False, save_plots=True):
    '''
    Function to plot k-distance graph, thus to see which distance, epsilon, to select based on the number of 
    neighbors we want to consider
    @params:
        dataset: dataset on which implement the DBSCAN
        k: number of neighbors to consider
        data_type: dataset type, if with dim reduced or not
        verbose (bool): print distances or not
        see_plots (bool): to display or not the plots
        save_plots (bool): to save or not the generated plots
    @returns: K-distance graph
    '''
    # Algorithm to find the nearest neighbors. DBSCAN is based on density of clusters but such parameter is retrieved
    # by considering the spatial disposition of the points and theur neighbors
    neigh = NearestNeighbors(n_neighbors=k)
    neigh = neigh.fit(dataset)
    distances, _ = neigh.kneighbors(dataset)
    if verbose:
        print("dataset: ", dataset)
        print("distances: ", distances.shape)
    distances = np.sort(distances[:, k-1]) # Retrieving the distances between the most distant points
    if verbose:
        print("sorted distances: ", distances)
        print("distances: ", distances.shape)
    if see_plots:
        plt.figure(figsize=(10, 10))
        plt.plot(distances)
        plt.xlabel('Points')
        plt.ylabel(f'{k}-th nearest neighbor distance')
        plt.title(f'DBSCAN K-distance Graph {data_type}')
        plt.show()
    if save_plots:
        os.makedirs('DBSCAN_clust_plots', exist_ok=True)
        plt.figure(figsize=(10, 10))
        plt.plot(distances)
        plt.xlabel('Points')
        plt.ylabel(f'{k}-th nearest neighbor distance')
        plt.title(f'DBSCAN K-distance Graph {data_type}')
        plt.savefig(f"DBSCAN_clust_plots/K_distance_Graph_{data_type}.png")

def DBSCAN_cluster(dataset, epsilon, min_samples, see_plots=True, verbose=False, contingencyTable=True, data_type="raw", save_plots=True):
    """
    Function to perform DBSCAN and visualise the results.
    @params:
        dataset: dataset on which perform DBSCAN
        epsilon: k-neighbor distance to retain. Chosen based on k-distance graph
        min_samples: Min number of samples to create a cluster
        see_plots (bool): To see or not plots
        verbose (bool): To see or not the results
        contingencyTable (bool): To see or not the contingency table
        data_type: type of dataset, with or without dim reduction
        save_plots (bool): to save or not the generated plots
    """
    # Run the algorithm on the dataset considering the parameters chosen from previous analysis
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    clusters = dbscan.fit_predict(dataset)
    if see_plots:
        if isinstance(dataset, np.ndarray): #Change into pandas dataframe if numpy
            dataset = pd.DataFrame(dataset)
        # Visualize the results
        # Define discrete colors for plots if wanted
        # colors = ['red', 'pink', 'lightblue', 'orange', 'green']
        # Create a ListedColormap for plot function
        # cmap = ListedColormap(colors)
        plt.figure(figsize=(10, 10))
        plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=clusters, cmap="viridis")
        plt.title(f'DBSCAN Clustering Results {data_type}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    if save_plots:
        os.makedirs('DBSCAN_clust_plots', exist_ok=True)
        if isinstance(dataset, np.ndarray): #Change into pandas dataframe if numpy
            dataset = pd.DataFrame(dataset)
        # Visualize the results
        # Define discrete colors for plots if wanted
        # colors = ['red', 'pink', 'lightblue', 'orange', 'green']
        # Create a ListedColormap for plot function
        # cmap = ListedColormap(colors)
        plt.figure(figsize=(10, 10))
        plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=clusters, cmap="viridis")
        plt.title(f'DBSCAN Clustering Results {data_type}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.savefig(f"DBSCAN_clust_plots/DBSCAN_results_{data_type}.png")
    if see_plots:
        if isinstance(dataset, np.ndarray): #Change into pandas dataframe if numpy
            dataset = pd.DataFrame(dataset)
        # Visualize the results with data labelled according to their landcover type
        # Define discrete colors
        colors = ['red', 'pink', 'lightblue', 'orange', 'green']
        # Create a ListedColormap for plot function
        cmap = ListedColormap(colors)
        plt.figure(figsize=(10, 10))
        plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=Y, cmap=cmap)
        plt.title(f'DBSCAN Clustering Results with {data_type} and samples colored according to their landcover type')
        # Create a manual legend
        unique_labels = np.unique(Y)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, label=f'{label}') for i, label in enumerate(unique_labels)]
        plt.legend(handles=legend_elements)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    if save_plots:
        os.makedirs('DBSCAN_clust_plots', exist_ok=True)
        if isinstance(dataset, np.ndarray): #Change into pandas dataframe if numpy
            dataset = pd.DataFrame(dataset)
        # Visualize the results with data labelled according to their landcover type
        # Define discrete colors
        colors = ['red', 'pink', 'lightblue', 'orange', 'green']
        # Create a ListedColormap for plot function
        cmap = ListedColormap(colors)
        plt.figure(figsize=(10, 10))
        plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=Y, cmap=cmap)
        plt.title(f'DBSCAN Clustering Results with {data_type} and samples colored according to their landcover type')
        # Create a manual legend
        unique_labels = np.unique(Y)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, label=f'{label}') for i, label in enumerate(unique_labels)]
        plt.legend(handles=legend_elements)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.savefig(f"DBSCAN_clust_plots/DBSCAN_results_{data_type}_landcover_labels.png")
    if verbose:
        # Print number of clusters and noise points
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        print(f'Number of clusters: {n_clusters} with {data_type}')
        print(f'Number of noise points: {n_noise} with {data_type}')
        print(f"Silhouette Coefficient with {data_type}: {metrics.silhouette_score(X, clusters):.3f}")
    if contingencyTable:
        # Display contingency table
        cross_tab = pd.crosstab(Y, clusters)
        print(f"contingency table DBSCAN with {data_type}")
        print(cross_tab)

if __name__ == '__main__':
    data = INSERT PATH OF YOUR DATASET LOCATION

    # Project-related inputs, to set
    parser = argparse.ArgumentParser(description="main")
    parser.add_argument('--random_seed', type=int, default=0, help = 'Random seed to be set.') #run with 0,1,2,3
    parser.add_argument('--data_path', type=str, default=data)
    parser.add_argument('--n_dims_pca', type=int, default=2, help = 'Number of dimensions for dimensionality reduction using PCA.')
    parser.add_argument('--var_threshold',type=int, default=0.10, help = 'Variance threshold to select features.')
    args = parser.parse_args()
    
    # Set seed
    random_seed=set_seed(random_seed=args.random_seed) 
    # Import dataset using raw string to avoid problems with syntax
    X,Y = data_preprocessing(dataset=args.data_path, verbose=False)
    # Dimensionality reduction PCA
    XPCA = pca_dim_reduction(dataset_original=args.data_path, dataset=X,n_dim=args.n_dims_pca, plots=False)
    # Dimensionality reduction VT
    XVT = variance_tresh_dim_reduction(dataset_original=data, variance_threshold=args.var_threshold, verbose=False)
    # DBSCAN without dimensionality reduction
    # Plot k-distance graph
    plot_k_distance_graph(X, k=5,data_type=f"no dim reduction, seed {args.random_seed}",see_plots=False, save_plots=True)
    # Retain distance indicated by the elbow, in this case is 2.75
    # This will be the minimum distance to identify the clusters, thus the Epsilon.
    # DBSCAN algorithm and results. Usually 2 * num_features, but here high dimensions, 
    # so I just use the min number of samples per landcover, which is 14.
    DBSCAN_cluster(X,epsilon=17,min_samples=14, see_plots=False, save_plots=True, verbose=True,contingencyTable=True,data_type=f"no dim reduction, seed {args.random_seed}")
    # DBSCAN with VT dimensionality reduction
    # Function to plot k-distance graph
    plot_k_distance_graph(XVT, k=5,data_type=f"VT dim reduction, seed {args.random_seed}")
    # Retain distance indicated by the elbow, in this case is 0.45
    # This will be the minimum distance to identify the clusters, thus the Epsilon.
    # Perform DBSCAN clustering with epsilon 0.5 and 4 as minimum number of samples, 
    # given that the usual 2 * num_features (7 variables retained) would be too much. Thus, I retained the value used for no dim reduced dimensions
    DBSCAN_cluster(XVT,epsilon=0.45,min_samples=4, see_plots=False, save_plots=True, verbose=True,contingencyTable=True,data_type=f"VT dim reduction, seed {args.random_seed}")
    # DBSCAN with PCA dimensionality reduction
    plot_k_distance_graph(XPCA, k=5, data_type=f"PCA dim reduction, seed {args.random_seed}")
    # Retain distance indicated by the elbow, in this case is 0.5, and 4 as minimum number of samples
    # Perform DBSCAN 
    DBSCAN_cluster(XPCA,epsilon=2.75,min_samples=4, see_plots=False, save_plots=True, verbose=True,contingencyTable=True,data_type=f"PCA dim reduction, seed {args.random_seed}")


