# Supervised and Unsupervised ML algorithms to understand the restoration potential of Acacia mangium
# K-means analysis

#region Library installation
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

def data_preprocessing(dataset_path, verbose=False):
    """
    This function is to preprocess the data, removing unsuited samples,
    removing the first column from the dataset of the variables (the first column has the labels of the data),
    and converting into integers the numeric values and scaling the whole dataset.
    @params:
        dataset_path (dataset, pandas or numpy): dataset to work on
        verbose (bool): specify if to print or not the steps and their result
    @returns: dataset divided into X and Y, where X is the dataset with features and samples, and Y the column with the labels of the samples.
    """
    df = pd.read_csv(dataset_path)
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
    all_top_contributions.to_csv(f'KMeans_top10_PC1_PC2_combined_contributions_with_seed_{args.random_seed}.csv')
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

def kmeans_parameters_study(data_kmeans, dataset, verbose=False, contingencyTable=True, see_plots=False, save_plots=True, data_type="raw"):
    """
    This function is to study the parameters of the kmeans and retrieve the contingency table
    @params:
        data_kmeans: KMeans() object
        dataset: dataset on which the kmeans was run
        verbose (bool): specify if to print or not the steps and their result
        contingencyTable (bool): specify to print or not the contingency table
        see_plots (bool): specify to see or not plots
        save_plots (bool): to save plots in folder
        data_type: if the dataset has reduced dimensions or not, and how
    @returns: contingency table and plots
    """
    # Recognize centers of the clusters
    if verbose:
        print(data_kmeans.cluster_centers_)
    # See inertia (the lower it is the better it is)
    if verbose:
        print(data_kmeans.inertia_)
    # Not necessary, but classification results:
    # Check quality of weak classification by the model
    labels = data_kmeans.labels_
    if verbose:
        print("Y: ", Y)
        print("labels: ", labels)
    # Calculate silhouette score
    silhouette_avg = silhouette_score(dataset, labels)
    if verbose:
        print(f"Silhouette Score {data_type} and {len(data_kmeans.cluster_centers_)} clusters: {silhouette_avg:.2f}")
    # Visualise plots with color of the predicted clusters
    if isinstance(dataset, np.ndarray): #Change into pandas dataframe if numpy
        dataset = pd.DataFrame(dataset)
    if see_plots:
        centers = data_kmeans.cluster_centers_  # Get the cluster centers
        plt.figure(figsize=(10,10))
        uniq = np.unique(labels)
        for i in uniq:
            plt.scatter(dataset.loc[labels == i, dataset.columns[0]], dataset.loc[labels == i, dataset.columns[1]], label=i)
        plt.scatter(centers[:, 0], centers[:, 1], marker="x", color='k') # plot the centers
        plt.legend()
        plt.title(f"K-means {len(data_kmeans.cluster_centers_)} clusters {data_type} ")
        plt.show()
    if  save_plots:
        os.makedirs('K_means_clust_plots', exist_ok=True)
        centers = data_kmeans.cluster_centers_  # Get the cluster centers
        plt.figure(figsize=(10,10))
        uniq = np.unique(labels)
        for i in uniq:
            plt.scatter(dataset.loc[labels == i, dataset.columns[0]], dataset.loc[labels == i, dataset.columns[1]], label=i)
        plt.scatter(centers[:, 0], centers[:, 1], marker="x", color='k')
        plt.legend()
        plt.title(f"K-means {len(data_kmeans.cluster_centers_)} clusters {data_type} ")
        plt.savefig(f"K_means_clust_plots/K_means_clust_{data_type}.png")
    # Plot with landcover labels colors
    if see_plots:
        if isinstance(dataset, np.ndarray):  #Change into pandas dataframe if numpy
            dataset = pd.DataFrame(dataset)
        # Visualize the results with data labelled according to their landcover type
        # Define discrete colors
        colors = ['red', 'pink', 'lightblue', 'orange', 'green']
        # Create a ListedColormap for plot function
        cmap = ListedColormap(colors)
        plt.figure(figsize=(10,10))
        plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=Y, cmap=cmap)
        plt.scatter(centers[:, 0], centers[:, 1], marker="x", color='k') # plot the centers
        # Create a manual legend
        unique_labels = np.unique(Y)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, label=f'{label}') for i, label in enumerate(unique_labels)]
        plt.legend(handles=legend_elements)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f"K-means {len(data_kmeans.cluster_centers_)} clusters and true labels colors with {data_type}")
        plt.show()
    if save_plots:
        os.makedirs('K_means_clust_plots', exist_ok=True)
        if isinstance(dataset, np.ndarray):  #Change into pandas dataframe if numpy
                dataset = pd.DataFrame(dataset)
        # Visualize the results with data labelled according to their landcover type
        # Define discrete colors
        colors = ['red', 'pink', 'lightblue', 'orange', 'green']
        # Create a ListedColormap for plot function
        cmap = ListedColormap(colors)
        plt.figure(figsize=(10,10))
        plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], c=Y, cmap=cmap)
        plt.scatter(centers[:, 0], centers[:, 1], marker="x", color='k') # plot the centers
        # Create a manual legend
        unique_labels = np.unique(Y)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, label=f'{label}') for i, label in enumerate(unique_labels)]
        plt.legend(handles=legend_elements)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f"K-means {len(data_kmeans.cluster_centers_)} clusters and true labels colors with {data_type}")
        plt.savefig(f"K_means_clust_plots/K_means_{len(data_kmeans.cluster_centers_)}_clust_with_{data_type}_landcover_labels.png")
    # Display contingency table
    cross_tab = pd.crosstab(Y, labels)
    if contingencyTable:
        print(f"contingency table k-means {len(data_kmeans.cluster_centers_)} clusters {data_type} \n",cross_tab)

def FindClustersNumber(data, see_plots=False, data_type="raw", save_plots=True):
    """"
    Function to calculate and see the optimal number of clusters through the elbow method.
    Note: If errors arise check the min 4 spaces in the second, third and fourth line of the for loop, then
    double press Enter to run it
    @params
        data: dataset to fit in kmeans algorithm
        see_plots(bool): Specify to see plots (for elbow method)
        save_plots (bool): to save plots in folder
        data_type: if the dataset has reduced dimensions or not, and how
    """
    # Create an empty list to store the data and run the Kmeans algorithm
    cs = []
    for i in range(1, 11):
     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0) # Random state can be changed
     kmeans.fit(data)
     cs.append(kmeans.inertia_)
    if see_plots:
        plt.plot(range(1, 11), cs)
        plt.title('The Elbow Method {}'.format(data_type))
        plt.xlabel('Number of clusters')
        plt.ylabel('CS')
        plt.show()
    if save_plots:
        os.makedirs('K_means_clust_plots', exist_ok=True)
        plt.plot(range(1, 11), cs)
        plt.title('The Elbow Method {}'.format(data_type))
        plt.xlabel('Number of clusters')
        plt.ylabel('CS')
        plt.savefig(f"K_means_clust_plots/K_means_elbow_{data_type}.png")

def kmeans_clustering(data, n_clusters, seed, verbose=False, contingencyTable=True, see_plots=False, save_plots=True, data_type="raw"):
    '''
    Funtion which englobes Kmeans algorithm and results visualisation
    @params:
        data: dataset
        n_clusters: number of clusters with which run the kmeans
        seed: random state
        verbose (bool): to print or not the results
        contingencyTable (bool): to print or not the contingency table
        see_plots (bool): to print or not the plots
        save_plots (bool): to save the plots or not
        data_type: dataset if with dim reduced or not
    @returns: Kmeans, results, plots
    '''
    # Set the K-mean algorithm on the dataset considering the chosen n of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    # Run algorithm on dataset
    kmeans.fit(data)
    print("RESULTS KMEANS {} CLUSTERS AND {} DATA".format(n_clusters, data_type))
    # K-Means parameters study (e.g., Silhouettes score, contingency table)
    kmeans_parameters_study(data_kmeans=kmeans, dataset=data, verbose=verbose, contingencyTable=contingencyTable, see_plots=see_plots, save_plots=save_plots, data_type=data_type)

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
    X,Y = data_preprocessing(dataset_path=args.data_path, verbose=False)
    # Dimensionality reduction PCA
    XPCA = pca_dim_reduction(dataset_original=args.data_path, dataset=X, n_dim=args.n_dims_pca, plots=True,verbose=True)
    # Dimensionality reduction VT
    XVT = variance_tresh_dim_reduction(dataset_original=data, variance_threshold=args.var_threshold, verbose=False)
    # Unsupervised ML: K-means
    kmeans_clustering(X, 2, seed=args.random_seed, verbose=True, contingencyTable=True, see_plots=True, data_type=f"no dim reduction, seed {args.random_seed}")
    # K-means with 5 clusters, given the 5 categories (landcover type)
    kmeans_clustering(X, 5, seed=args.random_seed, verbose=True, contingencyTable=True, see_plots=True, data_type=f"no dim reduction, seed {args.random_seed}")
    # Find the right number of clusters with the elbow method
    FindClustersNumber(X,see_plots=True,data_type=f"no dim reduction, seed {args.random_seed}") # Three clusters seems the right number to run the k-means
    # Run the k-means with three clusters
    kmeans_clustering(X, 3, seed=args.random_seed, verbose=True, contingencyTable=True, see_plots=True, data_type=f"no dim reduction, seed {args.random_seed}")
    # end analysis without dimensionality reduction
    # K-means with VT dimensionality reduction and 5 clusters
    kmeans_clustering(XVT, 5, seed=args.random_seed, verbose=True, contingencyTable=True, see_plots=True, data_type=f"VT dim reduction, seed {args.random_seed}")
    # K-means with VT dimensionality reduction and 3
    FindClustersNumber(XVT,see_plots=False, data_type=f"VT dim reduction, seed {args.random_seed}") # Find the right number of clusters with the elbow method
    # Two clusters seems the right number to run the k-means
    # K-means with VT dimensionality reduction and 2 clusters
    kmeans_clustering(XVT, 2, seed=args.random_seed, verbose=True, contingencyTable=True, see_plots=True, data_type=f"VT dim reduction, seed {args.random_seed}")
    # K-Means Clustering on PCA-args.random_seed data with 5 clusters
    kmeans_clustering(XPCA, 5, seed=args.random_seed, verbose=True, contingencyTable=True, see_plots=False, data_type=f"PCA reduction, seed {args.random_seed}")
    # Find the right number of clusters with the elbow method
    FindClustersNumber(XPCA,see_plots=True,data_type=f"PCA reduction, seed {args.random_seed}") # Find the right number of clusters with the elbow method
    # Three clusters seems the right number to run the k-means
    # K-means with dimensionality reduction and 3 clusters

    kmeans_clustering(XPCA, 3, seed=args.random_seed, verbose=True, contingencyTable=True, see_plots=True, data_type=f"PCA reduction, seed {args.random_seed}")

