# Supervised and Unsupervised ML algorithms to understand the restoration potential of Acacia mangium
# Code for Agglomerative Hierarchical Clustering

#region Library installation ####
# Unsupervised ML
# Install library for k-means in command prompt (Shell in Windows)
#pip install matplotlib
#pip install kneed
#pip install scikit-learn
#pip install pandas # for data processing
#pip install numpy # for linear algebra
#pip install seaborn # for statistical data visualization
#pip install scipy # for Agglomerative Hierarchical Clustering
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
#######Function with scipy library. Not used for analysis#####################
def Agg_hier_clust(dataset, method, data_type, max_d, Seeplots=True):  
    """
    Function to perform Agglomerative hierarchical clustering and obtain
    results and plots, and obtain the contingency table

    @params:
        dataset: dataset as input
        method: linkage method, choose between: average, ward, ...
        data_type: which dataset is being used (if with reduced dim or not)
        max_d: distance to cut the dendrogram      
        contingencyTable (bool):to print or not the contingency table
        verbose (bool): to print or not the results
        Seeplots (bool): to see or not the plots
    """
    # AHC with average linkage and no dim reduction
    Z = linkage(dataset, method=method)
    if Seeplots:
        # Plot the dendrogram
        plt.figure(figsize=(10, 10))
        dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15., show_contracted=True)
        plt.title(f'Dendrogram with {data_type} and {method} linkage')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plt.show()
    # Determine the clusters
    clusters = fcluster(Z, max_d, criterion='distance')
    if Seeplots:
        if isinstance(dataset, np.ndarray): #Change into pandas dataframe if numpy
            dataset = pd.DataFrame(dataset)
        # Plot the clustered data
        # Create the scatter plot using pandas iloc to access columns
        plt.figure(figsize=(10, 10))
        plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=clusters, cmap='viridis', s=50)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Hierarchical Agglomerative Clustering (HAC) with {data_type} and {method} linkage')
        plt.show()
        # Visualize the results with data labelled according to their landcover type
        # Define discrete colors
        colors = ['red', 'pink', 'lightblue', 'orange', 'green']
        # Create a ListedColormap for plot function
        cmap = ListedColormap(colors)
        plt.figure(figsize=(10, 10))
        plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=Y, cmap=cmap)
        plt.title(f'AHC Results with {data_type}, {method} linkage, and samples colored according to their landcover type')
        # Create a manual legend
        unique_labels = np.unique(Y)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, label=f'{label}') for i, label in enumerate(unique_labels)]
        plt.legend(handles=legend_elements)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Agglomerative Hierarchical Clustering (HAC) with {data_type} and {method} linkage')
        plt.show()
##############################################################################

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
    all_top_contributions.to_csv(f'AHC_top10_PC1_PC2_combined_contributions_with_seed_{args.random_seed}.csv')
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
    print(X_for_variance)
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

def aggl_clust_skl(dataset, method, data_type, max_d, n_clusters, contingencyTable=True, verbose=False, see_plots=False, save_plots=True):
    """
    Function to perform Agglomerative hierarchical clustering and obtain
    results and plots, and the contingency table. It uses the function 'AgglomerativeClustering' from scikit-learn.
    @params:
        dataset: dataset as input
        method: linkage method, choose between: average, ward, ...
        data_type: which dataset is being used (if with reduced dim or not)
        max_d: linkage distance threshold at or above which clusters will not be merged. It must be set to 'None' if n_cluster is specified
        n_clusters: number of clusters to retain. Directly connected to the distance at which cut the dendrogram. It must be set to 'None' if
        max_d is specified    
        contingencyTable (bool):to print or not the contingency table
        verbose (bool): to print or not the results
        see_plots (bool): to see or not the plots
        save_plots (bool): to save or not the generated plots
    """
    # Run the Agglomerative clustering algorithm and fit it with the dataset
    # Silhouette score by doing AHC with AgglomerativeClustering(). If max_d is not 'None', then n_clusters must be 'None',
    # and compute_full_tree must be 'True'.
    cluster = AgglomerativeClustering(linkage=method, distance_threshold=max_d,n_clusters=n_clusters, compute_full_tree=True)
    cluster.fit(dataset) 
    # Get the cluster labels
    labels = cluster.labels_
    # Calculate the Silhouette Score
    silhouette_avg = silhouette_score(dataset, labels)
    if verbose:
        # Print the Silhouette Score
        print(f"Silhouette Score AHC with {data_type} and {method} linkage: {silhouette_avg:.3f}")
    if contingencyTable:
        # contingency table
        # Display contingency table
        cross_tab = pd.crosstab(Y, cluster.labels_)
        print(f"contingency table AHC with {data_type} and {method} linkage")
        print(cross_tab)
    if see_plots:
        plt.figure(figsize=(10, 10))  
        plt.title("Dendrograms")  
        shc.dendrogram(shc.linkage(dataset, method=method))
        plt.show()
    if save_plots:
        os.makedirs('Agg_hier_clust_plots', exist_ok=True)
        plt.figure(figsize=(10, 10))  
        plt.title("Dendrograms")  
        shc.dendrogram(shc.linkage(dataset, method=method))
        plt.savefig(f"Agg_hier_clust_plots/Dendrogram_{data_type}_{method}.png")
    # Plots with landcover type labels and colors
    if see_plots:
        if isinstance(dataset, np.ndarray): #Change into pandas dataframe if numpy
            dataset = pd.DataFrame(dataset)
        # Plot the clustered data
        # Create the scatter plot using pandas iloc to access columns
        plt.figure(figsize=(10, 10))
        plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=labels, cmap='viridis', s=50)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.title(f'Hierarchical Agglomerative Clustering (HAC) with {data_type} and {method} linkage')
        plt.show()
    if save_plots:
        os.makedirs('Agg_hier_clust_plots', exist_ok=True)
        if isinstance(dataset, np.ndarray): #Change into pandas dataframe if numpy
            dataset = pd.DataFrame(dataset)
        # Plot the clustered data
        # Create the scatter plot using pandas iloc to access columns
        plt.figure(figsize=(10, 10))
        plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=labels, cmap='viridis', s=50)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Agglomerative Hierarchical Clustering with {data_type} and {method} linkage')
        plt.savefig(f"Agg_hier_clust_plots/Agg_Hier_clust_{data_type}_{method}.png")
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
        plt.title(f'AHC Results with {data_type}, {method} linkage, and samples colored according to their landcover type')
        # Create a manual legend
        unique_labels = np.unique(Y)
        legend_elements = [plt.Line2D([0], [0], marker='x', color='w', markerfacecolor=colors[i], markersize=10, label=f'{label}') for i, label in enumerate(unique_labels)]
        plt.legend(handles=legend_elements)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Agglomerative Hierarchical Clustering (HAC) with {data_type} and {method} linkage')
        plt.show()
    if save_plots:
        os.makedirs('Agg_hier_clust_plots', exist_ok=True)
        if isinstance(dataset, np.ndarray): #Change into pandas dataframe if numpy
            dataset = pd.DataFrame(dataset)
        # Visualize the results with data labelled according to their landcover type
        # Define discrete colors
        colors = ['red', 'pink', 'lightblue', 'orange', 'green']
        # Create a ListedColormap for plot function
        cmap = ListedColormap(colors)
        plt.figure(figsize=(10, 10))
        plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=Y, cmap=cmap)
        plt.title(f'AHC Results with {data_type}, {method} linkage, and samples colored according to their landcover type')
        # Create a manual legend
        unique_labels = np.unique(Y)
        legend_elements = [plt.Line2D([0], [0], marker='x', color='w', markerfacecolor=colors[i], markersize=10, label=f'{label}') for i, label in enumerate(unique_labels)]
        plt.legend(handles=legend_elements)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Agglomerative Hierarchical Clustering (HAC) with {data_type} and {method} linkage')
        plt.savefig(f"Agg_hier_clust_plots/Agg_Hier_clust_{data_type}_{method}_landcover_labels.png")

if __name__ == '__main__':
    data = INSERT PATH OF YOUR DATASET LOCATION
    # Project-related inputs, to set
    parser = argparse.ArgumentParser(description="main")
    parser.add_argument('--random_seed', type=int, default=3, help = 'Random seed to be set.') #run with 0,1,2,3
    parser.add_argument('--data_path', type=str, default=data)
    parser.add_argument('--n_dims_pca', type=int, default=2, help = 'Number of dimensions for dimensionality reduction using PCA.')
    parser.add_argument('--var_threshold',type=int, default=0.10, help = 'Variance threshold to select features.')
    args = parser.parse_args()
    
    # Set seed
    random_seed=set_seed(random_seed=args.random_seed) 
    # Import dataset using raw string to avoid problems with syntax
    X,Y = data_preprocessing(dataset=args.data_path, verbose=False)
    # Dimensionality reduction PCA
    XPCA = pca_dim_reduction(dataset_original=args.data_path, dataset=X, n_dim=args.n_dims_pca, plots=False)
    # Dimensionality reduction VT
    XVT = variance_tresh_dim_reduction(dataset_original=data, variance_threshold=args.var_threshold, verbose=False)
    # AHC without dimensionality reduction and average linkage
    # Average linkage uses the average distance between all the points in each cluster
    aggl_clust_skl(X,method="average",contingencyTable=True,n_clusters=2,verbose=True,data_type=f"no dim reduction, seed {args.random_seed}",max_d=None, see_plots=False, save_plots=True)
    # AHC without dimensionality reduction and Ward's linkage
    # Ward's linkage uses as distance the increase in sum of squares when 2 clusters are merged together
    aggl_clust_skl(X,method="ward",contingencyTable=True,verbose=True,data_type=f"no dim reduction, seed {args.random_seed}",max_d=None,n_clusters=2, see_plots=False, save_plots=True)
    # AHC with VT dimensionality reduction and average linkage
    aggl_clust_skl(XVT,method="average",contingencyTable=True,verbose=True,data_type=f"VT dim reduction, seed {args.random_seed}",max_d=None,n_clusters=2, see_plots=False, save_plots=True)
    # AHC with VT dimensionality reduction and Ward's linkage
    aggl_clust_skl(XVT,method="ward",contingencyTable=True,verbose=True,data_type=f"VT dim reduction, seed {args.random_seed}",max_d=None,n_clusters=2, see_plots=False, save_plots=True)
    # AHC with PCA dimensionality reduction and average linkage
    aggl_clust_skl(XPCA,method="average",contingencyTable=True,verbose=True,data_type=f"PCA dim reduction, seed {args.random_seed}",max_d=None,n_clusters=2, see_plots=False, save_plots=True)
    # AHC with PCA dimensionality reduction and Ward's linkage

    aggl_clust_skl(XPCA,method="ward",contingencyTable=True,verbose=True,data_type=f"PCA dim reduction, seed {args.random_seed}",max_d=None,n_clusters=2, see_plots=False, save_plots=True)

