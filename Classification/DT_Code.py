# Supervised and Unsupervised ML algorithms to understand the restoration potential of Acacia mangium
# Decision Tree analysis

#region Library installation ####
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
from sklearn.tree import plot_tree
#endregion

def set_seed(random_seed=0):
    np.random.seed(random_seed)
    return random_seed

def data_preprocessing(dataset_path, test_size, seed, verbose=False):
    """
    This function is to preprocess the data, removing unsuited samples,
    removing the first column from the dataset of the variables (the first column has the labels of the data),
    and converting into integers the numeric values and scaling the whole dataset.
    @params:
        dataset (dataset, pandas or numpy): dataset to work on
        test_size: proportion to split the dataset into training and test set
        seed: random state from which start to split the dataset
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
    # Convert categorical variable into integers ordering them
    map_landcover_label = {"Grassland": 0, "2 years old": 1, "10 years old": 2, "24 years old": 3, "Remnant": 4}
    Y = [map_landcover_label[lab] for lab in Y]
    # le = LabelEncoder() # This 2 lines to implement to change into integers the labels without specifications
    # Y = le.fit_transform(Y)
    if verbose:
        print("Y: ", Y)
    # Datasets separation
    # Split the data into training and test sets
    if verbose:
        print("X: ", X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    if verbose:
        print("X_train: ", X_train)
    # Feature scaling training set
    ms_ss = StandardScaler()
    X_train_transformed_ss = ms_ss.fit_transform(X_train)
    X_train_ss = pd.DataFrame(X_train_transformed_ss, columns=X_train.columns)
    if verbose:
        print("X_train after scaling: ", X_train_ss)
    # Feature scaling test set
    X_test_transformed_ss = ms_ss.transform(X_test)
    X_test_ss = pd.DataFrame(X_test_transformed_ss, columns=X_test.columns)
    if verbose:
        print("X_test after scaling: ", X_test_ss)

    ms_mm = MinMaxScaler()
    X_train_transformed_mm = ms_mm.fit_transform(X_train)
    X_train_mm = pd.DataFrame(X_train_transformed_mm, columns=X_train.columns)
    # Feature scaling test set
    X_test_transformed_mm = ms_mm.transform(X_test)
    X_test_mm = pd.DataFrame(X_test_transformed_mm, columns=X_test.columns)

    return X, Y, X_train_ss, X_test_ss, Y_train, Y_test, X_train_mm, X_test_mm

def pca_dim_reduction(dataset_original, data_train, data_test, n_dim,verbose=False, see_plots=False):
    """
    This function is to implement feature extraction by reducing the X dataset dimensions through PCA analysis, retaining 2 PCs.
    @params:
        data_train: dataset for training
        data_test: dataset for test
        n_dim: number of dimension to retain
        see_plots (bool): specify if to see plots or not
    @returns: dataset with dimensions reduced to 2.
    """
    df = pd.read_csv(dataset_original)
    names_to_remove = ['I2NB', 'O2NB', 'O4NA', 'R1SA', 'R1SB']
    df = df[~df['Sample_ID'].isin(names_to_remove)]
    # Now remove the first column, which is "Sample_ID"
    df = df.iloc[:, 1:]
    X = df.iloc[:,1:] # select all rows and all the columns from the third, excluding the first 2 categorical
    pca = PCA(n_dim) # Convert dataset in n dimensions
    XPCA_train = pca.fit_transform(data_train)
    # Check the variance of the components
    var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
    lbls = [str(x) for x in range(1,len(var)+1)]
    if see_plots:
        plt.figure(figsize=(10,10))
        plt.bar(x=range(1,len(var)+1), height = var, tick_label = lbls)
        plt.show()
    
    # Get and display loadings (Original Variable Contributions)
    # The pca.components_ attribute contains the loadings, each row is a PC, each column is an original feature
    loadings = pd.DataFrame(pca.components_,
                            columns=X.columns, # Use the correctly determined feature names
                            index=[f'PC{i+1}' for i in range(pca.n_components_)])
    if verbose:
        print("--- PCA Explained Variance with {args.random_seed}---")
        # Explained Variance Ratio for PC1
        explained_variance_pc1 = pca.explained_variance_ratio_[0]
        print(f"Explained Variance Ratio by PC1 with {args.random_seed}: {explained_variance_pc1:.4f}")
        # Explained Variance Ratio for PC2
        explained_variance_pc2 = pca.explained_variance_ratio_[1]
        print(f"Explained Variance Ratio by PC2 with {args.random_seed}: {explained_variance_pc2:.4f}")
        print("\n--- Principal Component Loadings (Original Variable Contributions) ---")
        print("Each row is a PC, each column is an original feature.")
        print("Values indicate the weight/contribution of each original feature to that PC.")
        print("Larger absolute values signify a stronger contribution.")
        print(loadings)
        # Top 10 variables contribution to the first 2 PCs
        print("\n--- Contributions of Original Variables for the First 2 PCs ---")
        if n_dim >= 1:
            print(f"\nPC1 Loadings (Contributions) with {args.random_seed}:")
            pc1_contributions = loadings.loc['PC1'].sort_values(key=abs, ascending=False).head(10)
            print(pc1_contributions) # Sort by absolute value for contribution, False will sort ascending, True will sort descending
        if n_dim >= 2:
            print(f"\nPC2 Loadings (Contributions) with {args.random_seed}:")
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
    all_top_contributions.to_csv(f'KNN_top10_PC1_PC2_combined_contributions_with_seed_{args.random_seed}.csv')
    print(f"\nSaved combined top 10 contributions for PC1 and PC2")

    # Convert test dataset in 2 dimensions
    # Apply only transform to avoid another different fitting for the test set
    XPCA_test = pca.transform(data_test)
    # Check the variance of the components
    var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
    lbls = [str(x) for x in range(1,len(var)+1)]
    if see_plots:
        plt.figure(figsize=(10,10))
        plt.bar(x=range(1,len(var)+1), height = var, tick_label = lbls)
        plt.show()
    return XPCA_train, XPCA_test

def variance_tresh_dim_reduction(dataset_training_mm, dataset_test_mm, variance_threshold, seed, verbose=False):
    """
    This function is to implement feature selection by reducing the X dataset dimensions through Variance Treshold analysis, retaining 2 features.
    @params:
        dataset_training_mm: training dataset already scaled
        dataset_test_mm: test dataset already scaled
        variance_threshold: threshold to select the feature(s)
        verbose (bool): specify if to print or not the steps and their result
    @returns: XVT_train and XVT_test datasets with dimensions reduced.
    """
    # Study variance of the scaled dataset
    variance = np.var(dataset_training_mm)
    if verbose:
        print("Variance:", variance)
    # Save to Excel file to analyse better and explore the treshold to retain 2, 3, or n number of variables.
    variance.to_csv(f"VarianceFeaturesClassification_{seed}.csv")
    # Set the threshold 
    VT = VarianceThreshold(threshold=variance_threshold) 
    # Application of feature selection on training dataset scaled through MinMaxScaler()
    XVT_train_transformed = VT.fit_transform(dataset_training_mm)
    if verbose:
        print("XVT_train shape:", XVT_train_transformed.shape)
    # Apply only transform to avoid another different fitting for the test set
    XVT_test_transformed = VT.transform(dataset_test_mm)
    if verbose:
        print("XVT_test shape:", XVT_test_transformed.shape)
    # Get the names of the features that were *not* removed by VarianceThreshold
    selected_features = dataset_training_mm.columns[VT.get_support()]
    XVT_train = pd.DataFrame(XVT_train_transformed, columns=selected_features)
    XVT_test = pd.DataFrame(XVT_test_transformed, columns=selected_features)
    if verbose:
        print("Shape of training data after Variance Threshold:", XVT_train.shape)
        print("Shape of testing data after Variance Threshold:", XVT_test.shape)
        print("XVT_train: ", XVT_train)
        print("XVT_test: ", XVT_test)
    return XVT_train, XVT_test

def pearson_corr_dim_reduction(labels, data_train, data_test, corr_threshold_with_target,corr_threshold_between_feat, seed, verbose=False, see_plots=True):

    """
    This function is to implement feature selection by reducing the X dataset dimensions through Variance Treshold analysis, retaining 2 features.
    @params:
        labels: target variable
        data_train: dataset of variables for training
        data_test: dataset used for test
        corr_threshold_with_target: value of the r to filter and select the features of interest after analysis
        of correlation between variables and target
        corr_threshold_between_feat: value of the r to filter and select the features of interest after analysis \
        of correlations between variables
        verbose (bool): specify if to print or not the steps and their result
    @returns: X dataset with dimensions reduced: XP_train and XP_test
    """
    # Convert into pandas dataframe
    #print("original_dataset: ", original_dataset)
    #original_dataset=pd.DataFrame(original_dataset)
    #print("original_dataset: ", original_dataset)
    #print("labels: ", labels)
    labels = pd.Series(labels)
    # Implement regression between variables and target from the training dataset
    regression_table = data_train.corrwith(labels)
    if verbose:
        print ("Correlation between variables and target",regression_table)
    if see_plots:
        correlation_df_target = pd.DataFrame(regression_table, columns=['Correlation_with_Y'])
        sns.heatmap(correlation_df_target, annot=True, cmap='coolwarm', fmt=".2f") # Analyse correlation with target
        plt.title("Correlation of Features with Target Variable")
        plt.show()
    # Save the correlation matrix to a CSV file
    regression_table.to_csv(f"corr_matrix_original_dataset_{seed}.csv", index=True)
    # Select the variables based on chosed trheshold
    relevant_features_target = regression_table[abs(regression_table) >= corr_threshold_with_target].index
    if verbose:
        print("\nFeatures selected based on correlation with target (threshold >= {}):\n".format(corr_threshold_with_target), relevant_features_target)
    # Filter the training data to include only the selected features
    XP_train = data_train[relevant_features_target]
    # Implement correlation between the selected variables in the training set
    regression_variables = XP_train.corr()
    if verbose:
        print("\nCorrelation matrix between selected features in the training set:\n", regression_variables)
    if see_plots:
        plt.figure(figsize=(10, 8))
        sns.heatmap(regression_variables, annot=True, cmap='coolwarm', fmt=".2f") # Analyze and filter the features correlated to each other
        plt.title("Correlation Matrix of Selected Features in Training Set")
        plt.show()
    # Save the correlation matrix to a CSV file
    regression_variables.to_csv(f"corr_matrix_between_features_{seed}.csv", index=True)
    # Remove highly correlated columns
    columns_to_keep = np.full((regression_variables.shape[0],), True, dtype=bool)
    for i in range(regression_variables.shape[0]):
        for j in range(i + 1, regression_variables.shape[0]):
            if abs(regression_variables.iloc[i, j]) >= corr_threshold_between_feat:  # Select threshold based on absolute value
                if columns_to_keep[j]:
                    columns_to_keep[j] = False
    selected_features_pairwise = regression_variables.columns[columns_to_keep]
    if verbose:
        print("\nFeatures remaining after removing highly correlated features (threshold >= {}):\n".format(corr_threshold_between_feat), selected_features_pairwise)
    # Filter the columns
    XP_train = XP_train[selected_features_pairwise]
    XP_test = data_test[selected_features_pairwise]
    return XP_train, XP_test

def Decision_tree_crossval(dataset_train, labels_train,random_state,data_type,verbose=True):
    """
    This function is to train the decision tree and choose the best hyperparameters combination.
    @params:
        dataset_train: training dataset
        labels_train: labels of the training dataset
        random_state: seed from which start the computations
        data_type: type of dataset (if of reduced dimensions or not)
        verbose (bool): to print the results
    """
    # Hyperparameter to fine tune
    param_grid = {
        'max_depth': range(1, 10, 1),
        'min_samples_leaf': range(2, 20, 2),
        'min_samples_split': range(2, 20, 2),
        'max_leaf_nodes': range (2,10,1),
        'criterion': ["entropy", "gini"]
    }
    tree = DecisionTreeClassifier(random_state=random_state)
    # GridSearchCV
    grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5, verbose=True)
    grid_search.fit(dataset_train, labels_train)
    if verbose:
        print(f"best accuracy of {data_type}", grid_search.best_score_)
        print(grid_search.best_estimator_)
        print("if not displayed, use default values of the non-printed hyperparameter:")
        print("max_depth: None, min_samples_leaf: 1, min_samples_split: 2, random_state: None, max_leaf_nodes: None, criterion: 'gini'")

def Decision_Tree_classification(dataset_train, labels_train, dataset_test,labels_test,
                                  max_depth, min_samples_leaf, max_leaf_nodes,min_samples_split,criterion, random_state, 
                                  data_type,confusion_mat=True, verbose=True, see_plots=False, save_plots=True,accuracy_per_class=False):
    """
    This function implements the decision tree classification.
    @params:
        dataset_train: training dataset
        labels_train: labels of the training dataset
        dataset_test: test dataset
        labels_test: labels of the test dataset
        max_depth: maximum lenght of the tree from root to leaf
        min_sample_leaf: minimum number of samples to stand in a leaf
        max_leaf_nodes: max number of leaves
        min_samples_split: minimum number of samples to split an internal node
        criterion: criterion to split the data, it can be "entropy" or "gini"
        random_state: seed from which start the computations
        data_type: type of dataset, if of reduced dimensions or not
        confusion_mat (bool): to print the confusion matrix
        verbose (bool): to print the results
        see_plots (bool): to visualise the plots
        save_plots (bool): to save the plots 
    """
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes, 
                                 criterion=criterion,min_samples_split=min_samples_split, random_state=random_state)
    # Train Decision Tree Classifer
    clf = clf.fit(dataset_train,labels_train)
    #Predict the response for test dataset
    Y_pred = clf.predict(dataset_test)
    if accuracy_per_class:
        unique_labels = np.unique(labels_test)
        # Accuracy for each class
        # Get the confusion matrix
        cm = confusion_matrix(labels_test, Y_pred)
        # Normalize the diagonal entries
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # The diagonal entries are the accuracies of each class
        Acc_class = {
            "Class": [unique_labels], 
            "Accuracy": [cm.diagonal()]
            }
        Acc_per_class=pd.DataFrame(Acc_class)
        print(f"Accuracy table DT  with {data_type}",Acc_per_class)
    # Confusion matrix
    if confusion_mat:
        # Display confusion matrix 
        cross_tab = pd.crosstab(labels_test, Y_pred)
        print(f"confusion matrix DT with {data_type}")
        print(cross_tab)
    # Model metrics
    if verbose:
        print("Accuracy:",metrics.accuracy_score(labels_test, Y_pred))
        # Model precision
        print(f"Precision with {data_type} (micro):", metrics.precision_score(labels_test, Y_pred, average='micro'))
        print(f"Precision with {data_type} (macro):", metrics.precision_score(labels_test, Y_pred, average='macro'))
        print(f"Precision with {data_type} (weighted):", metrics.precision_score(labels_test, Y_pred, average='weighted'))
        print(f"Precision with {data_type} (per class):", metrics.precision_score(labels_test, Y_pred, average=None))
        # Model Recall
        print(f"Recall with {data_type} (micro):",metrics.recall_score(labels_test, Y_pred, average='micro'))
        print(f"Recall with {data_type} (macro):",metrics.recall_score(labels_test, Y_pred, average='macro'))
        print(f"Recall with {data_type} (weighted):",metrics.recall_score(labels_test, Y_pred, average='weighted'))
        print(f"Recall with {data_type} (per class):",metrics.recall_score(labels_test, Y_pred, average=None))  
    # Plots
    if "PCA dim reduction" in data_type:
        if isinstance(dataset_train, np.ndarray):
            dataset_train = pd.DataFrame(dataset_train)
        else:
            dataset_train = dataset_train
    if see_plots:
        map_landcover_label = ("Grassland", "2 years old", "10 years old", "24 years old", "Remnant")
        plt.figure(figsize=(20, 12))
        plot_tree(clf,feature_names=dataset_train.columns, class_names= map_landcover_label)
        plt.title(f"DT with {data_type} and training data")
        plt.show()
    if save_plots:
        os.makedirs('DT_plots', exist_ok=True)
        map_landcover_label = ("Grassland", "2 years old", "10 years old", "24 years old", "Remnant")
        plt.figure(figsize=(20, 12))
        plot_tree(clf,feature_names=dataset_train.columns, class_names= map_landcover_label)
        plt.title(f"DT with {data_type} and training data")
        plt.savefig(f"DT_plots/DT_with_{data_type}.png")

def data_preprocessing_10_24(dataset_10_24,test_size,seed,verbose=False):
    """
    This function is to preprocess the data, removing unsuited samples,
    removing the first column from the dataset of the variables (the first column has the labels of the data),
    and converting into integers the numeric values and scaling the whole dataset.
    @params:
        dataset_10_24 (dataset, pandas or numpy): dataset with the same label for the 10 and 24 year old plantation to work on
        test_size: proportion to split the dataset into training and test set
        seed: random state from which start to split the dataset
        verbose (bool): specify if to print or not the steps and their result
    @returns: dataset divided into X and Y, where X is the dataset with features and samples, and Y the column with the labels of the samples.
    """
    df_10_24 = pd.read_csv(dataset_10_24)
    # Exclusion of samples I2NB, O2NB, O4NA, R1SA, R1SB because bacteria and fungi data with
    # not enough sequencing depth
    # Use boolean indexing to filter out the rows
    # Specify names to remove
    names_to_remove = ['I2NB', 'O2NB', 'O4NA', 'R1SA', 'R1SB']
    df_10_24 = df_10_24[~df_10_24['Sample_ID'].isin(names_to_remove)]
    if verbose:
        print("df_10_24.shape: ", df_10_24.shape)
        print("df_10_24: ", df_10_24)
    # Now remove the first column, which is "Sample_ID"
    df_10_24 = df_10_24.iloc[:, 1:]
    if verbose: # Print the filtered DataFrame
        print(df_10_24)
        print("df_10_24.shape: ", df_10_24.shape)
    # Python should already see that all columns have numeric data besides the first two
    # Statistical summary of the dataset
    # print(df.describe())
    # Declare feature vector and target variable
    # Now is treated as numpy df
    X_10_24 = df_10_24.iloc[:,1:] # select all rows and all the columns from the third, excluding the first 2 categorical
    if verbose:
        print("X_10_24.shape: ", X_10_24.shape)
        print("X_10_24: ", X_10_24)
    Y_10_24 = df_10_24['Landcover']
    if verbose:
        print("Y_10_24.shape: ", Y_10_24.shape)
        print("Y_10_24: ", Y_10_24)
    # Convert categorical variable into integers ordering them
    map_landcover_label = {"Grassland": 0, "2 years old": 1, "10_24 years old": 2, "Remnant": 3}
    Y_10_24 = [map_landcover_label[lab] for lab in Y_10_24]
    # le = LabelEncoder() # This 2 lines to implement to change into integers the labels without specifications
    # Y_10_24 = le.fit_transform(Y_10_24)
    if verbose:
        print("Y_10_24: ", Y_10_24)
    # Datasets separation
    # Split the data into training and test sets
    if verbose:
        print("X_10_24: ", X_10_24)
    X_10_24_train, X_10_24_test, Y_10_24_train, Y_10_24_test = train_test_split(X_10_24, Y_10_24,test_size=test_size,random_state=seed)
    if verbose:
        print("X_10_24_train: ", X_10_24_train)
    # Feature scaling training set
    ms = StandardScaler()
    X_10_24_train_transformed = ms.fit_transform(X_10_24_train)
    X_10_24_train = pd.DataFrame(X_10_24_train_transformed, columns=X_10_24_train.columns)
    if verbose:
        print("X_10_24_train after scaling: ", X_10_24_train)
    # Feature scaling test set
    X_10_24_test_transformed = ms.transform(X_10_24_test)
    X_10_24_test = pd.DataFrame(X_10_24_test_transformed,columns=X_10_24_test.columns)
    if verbose:
        print("X_10_24_test after scaling: ", X_10_24_test)
    return X_10_24,Y_10_24,X_10_24_train,X_10_24_test,Y_10_24_train,Y_10_24_test

def data_preprocessing_24_R(dataset_24_R,test_size,seed,verbose=False):
    """
    This function is to preprocess the data, removing unsuited samples,
    removing the first column from the dataset of the variables (the first column has the labels of the data),
    and converting into integers the numeric values and scaling the whole dataset.
    @params:
        dataset_24_R (dataset, pandas or numpy): dataset with same label for 24 year old plantation and remnant forest to work on
        test_size: proportion to split the dataset into training and test set
        seed: random state from which start to split the dataset
        verbose (bool): specify if to print or not the steps and their result
    @returns: dataset divided into X and Y, where X is the dataset with features and samples, and Y the column with the labels of the samples.
    """
    df_24_R = pd.read_csv(dataset_24_R)
    # Exclusion of samples I2NB, O2NB, O4NA, R1SA, R1SB because bacteria and fungi data with
    # not enough sequencing depth
    # Use boolean indexing to filter out the rows
    # Specify names to remove
    names_to_remove = ['I2NB', 'O2NB', 'O4NA', 'R1SA', 'R1SB']
    df_24_R = df_24_R[~df_24_R['Sample_ID'].isin(names_to_remove)]
    if verbose:
        print("df_24_R.shape: ", df_24_R.shape)
        print("df_24_R: ", df_24_R)
    # Now remove the first column, which is "Sample_ID"
    df_24_R = df_24_R.iloc[:, 1:]
    if verbose: # Print the filtered DataFrame
        print(df_24_R)
        print("df_24_R.shape: ", df_24_R.shape)
    # Python should already see that all columns have numeric data besides the first two
    # Statistical summary of the dataset
    # print(df.describe())
    # Declare feature vector and target variable
    # Now is treated as numpy df
    X_24_R = df_24_R.iloc[:,1:] # select all rows and all the columns from the third, excluding the first 2 categorical
    if verbose:
        print("X_24_R.shape: ", X_24_R.shape)
        print("X_24_R: ", X_24_R)
    Y_24_R = df_24_R['Landcover']
    if verbose:
        print("Y_24_R.shape: ", Y_24_R.shape)
        print("Y_24_R: ", Y_24_R)
    # Convert categorical variable into integers ordering them
    map_landcover_label = {"Grassland": 0, "2 years old": 1, "10 years old": 2, "24_R": 3}
    Y_24_R = [map_landcover_label[lab] for lab in Y_24_R]
    # le = LabelEncoder() # This 2 lines to implement to change into integers the labels without specifications
    # Y_24_R = le.fit_transform(Y_24_R)
    if verbose:
        print("Y_24_R: ", Y_24_R)
    # Datasets separation
    # Split the data into training and test sets
    if verbose:
        print("X_24_R: ", X_24_R)
    X_24_R_train, X_24_R_test, Y_24_R_train, Y_24_R_test = train_test_split(X_24_R, Y_24_R,test_size=test_size,random_state=seed)
    if verbose:
        print("X_24_R_train: ", X_24_R_train)
    # Feature scaling training set
    ms = StandardScaler()
    X_24_R_train_transformed = ms.fit_transform(X_24_R_train)
    X_24_R_train = pd.DataFrame(X_24_R_train_transformed, columns=X_24_R_train.columns)
    if verbose:
        print("X_24_R_train after scaling: ", X_24_R_train)
    # Feature scaling test set
    X_24_R_test_transformed = ms.transform(X_24_R_test)
    X_24_R_test = pd.DataFrame(X_24_R_test_transformed,columns=X_24_R_test.columns)
    if verbose:
        print("X_24_R_test after scaling: ", X_24_R_test)
    return X_24_R,Y_24_R,X_24_R_train,X_24_R_test,Y_24_R_train,Y_24_R_test

def data_preprocessing_10_R(dataset_10_R,test_size,seed,verbose=False):
    """
    This function is to preprocess the data, removing unsuited samples,
    removing the first column from the dataset of the variables (the first column has the labels of the data),
    and converting into integers the numeric values and scaling the whole dataset.
    @params:
        dataset_10_R (dataset, pandas or numpy): dataset with same label for 10 year old plantation and remnant forest to work on
        test_size: proportion to split the dataset into training and test set
        seed: random state from which start to split the dataset
        verbose (bool): specify if to print or not the steps and their result
    @returns: dataset divided into X and Y, where X is the dataset with features and samples, and Y the column with the labels of the samples.
    """
    df_10_R = pd.read_csv(dataset_10_R)
    # Exclusion of samples I2NB, O2NB, O4NA, R1SA, R1SB because bacteria and fungi data with
    # not enough sequencing depth
    # Use boolean indexing to filter out the rows
    # Specify names to remove
    names_to_remove = ['I2NB', 'O2NB', 'O4NA', 'R1SA', 'R1SB']
    df_10_R = df_10_R[~df_10_R['Sample_ID'].isin(names_to_remove)]
    if verbose:
        print("df_10_R.shape: ", df_10_R.shape)
        print("df_10_R: ", df_10_R)
    # Now remove the first column, which is "Sample_ID"
    df_10_R = df_10_R.iloc[:, 1:]
    if verbose: # Print the filtered DataFrame
        print(df_10_R)
        print("df_10_R.shape: ", df_10_R.shape)
    # Python should already see that all columns have numeric data besides the first two
    # Statistical summary of the dataset
    # print(df.describe())
    # Declare feature vector and target variable
    # Now is treated as numpy df
    X_10_R = df_10_R.iloc[:,1:] # select all rows and all the columns from the third, excluding the first 2 categorical
    if verbose:
        print("X_10_R.shape: ", X_10_R.shape)
        print("X_10_R: ", X_10_R)
    Y_10_R = df_10_R['Landcover']
    if verbose:
        print("Y_10_R.shape: ", Y_10_R.shape)
        print("Y_10_R: ", Y_10_R)
    # Convert categorical variable into integers ordering them
    map_landcover_label = {"Grassland": 0, "2 years old": 1, "10_R": 2, "24 years old": 3}
    Y_10_R = [map_landcover_label[lab] for lab in Y_10_R]
    # le = LabelEncoder() # This 2 lines to implement to change into integers the labels without specifications
    # Y_10_R = le.fit_transform(Y_10_R)
    if verbose:
        print("Y_10_R: ", Y_10_R)
    # Datasets separation
    # Split the data into training and test sets
    if verbose:
        print("X_10_R: ", X_10_R)
    X_10_R_train, X_10_R_test, Y_10_R_train, Y_10_R_test = train_test_split(X_10_R, Y_10_R,test_size=test_size,random_state=seed)
    if verbose:
        print("X_10_R_train: ", X_10_R_train)
    # Feature scaling training set
    ms = StandardScaler()
    X_10_R_train_transformed = ms.fit_transform(X_10_R_train)
    X_10_R_train = pd.DataFrame(X_10_R_train_transformed, columns=X_10_R_train.columns)
    if verbose:
        print("X_10_R_train after scaling: ", X_10_R_train)
    # Feature scaling test set
    X_10_R_test_transformed = ms.transform(X_10_R_test)
    X_10_R_test = pd.DataFrame(X_10_R_test_transformed,columns=X_10_R_test.columns)
    if verbose:
        print("X_10_R_test after scaling: ", X_10_R_test)
    return X_10_R,Y_10_R,X_10_R_train,X_10_R_test,Y_10_R_train,Y_10_R_test

def data_preprocessing_G_2(dataset_G_2,test_size,seed,verbose=False):
    """
    This function is to preprocess the data, removing unsuited samples,
    removing the first column from the dataset of the variables (the first column has the labels of the data),
    and converting into integers the numeric values and scaling the whole dataset.
    @params:
        dataset_G_2 (dataset, pandas or numpy): dataset with same label for 10 year old plantation and remnant forest to work on
        test_size: proportion to split the dataset into training and test set
        seed: random state from which start to split the dataset
        verbose (bool): specify if to print or not the steps and their result
    @returns: dataset divided into X and Y, where X is the dataset with features and samples, and Y the column with the labels of the samples.
    """
    df_G_2 = pd.read_csv(dataset_G_2)
    # Exclusion of samples I2NB, O2NB, O4NA, R1SA, R1SB because bacteria and fungi data with
    # not enough sequencing depth
    # Use boolean indexing to filter out the rows
    # Specify names to remove
    names_to_remove = ['I2NB', 'O2NB', 'O4NA', 'R1SA', 'R1SB']
    df_G_2 = df_G_2[~df_G_2['Sample_ID'].isin(names_to_remove)]
    if verbose:
        print("df_G_2.shape: ", df_G_2.shape)
        print("df_G_2: ", df_G_2)
    # Now remove the first column, which is "Sample_ID"
    df_G_2 = df_G_2.iloc[:, 1:]
    if verbose: # Print the filtered DataFrame
        print(df_G_2)
        print("df_G_2.shape: ", df_G_2.shape)
    # Python should already see that all columns have numeric data besides the first two
    # Statistical summary of the dataset
    # print(df.describe())
    # Declare feature vector and target variable
    # Now is treated as numpy df
    X_G_2 = df_G_2.iloc[:,1:] # select all rows and all the columns from the third, excluding the first 2 categorical
    if verbose:
        print("X_G_2.shape: ", X_G_2.shape)
        print("X_G_2: ", X_G_2)
    Y_G_2 = df_G_2['Landcover']
    if verbose:
        print("Y_G_2.shape: ", Y_G_2.shape)
        print("Y_G_2: ", Y_G_2)
    # Convert categorical variable into integers ordering them
    map_landcover_label = {"G_2": 0, "10 years old": 1, "24 years old": 2, "Remnant": 3}
    Y_G_2 = [map_landcover_label[lab] for lab in Y_G_2]
    # le = LabelEncoder() # This 2 lines to implement to change into integers the labels without specifications
    # Y_10_R = le.fit_transform(Y_10_R)
    if verbose:
        print("Y_G_2: ", Y_G_2)
    # Datasets separation
    # Split the data into training and test sets
    if verbose:
        print("X_G_2: ", X_G_2)
    X_G_2_train, X_G_2_test, Y_G_2_train, Y_G_2_test = train_test_split(X_G_2, Y_G_2,test_size=test_size,random_state=seed)
    if verbose:
        print("X_G_2_train: ", X_G_2_train)
    # Feature scaling training set
    ms = StandardScaler()
    X_G_2_train_transformed = ms.fit_transform(X_G_2_train)
    X_G_2_train = pd.DataFrame(X_G_2_train_transformed, columns=X_G_2_train.columns)
    if verbose:
        print("X_G_2_train after scaling: ", X_G_2_train)
    # Feature scaling test set
    X_G_2_test_transformed = ms.transform(X_G_2_test)
    X_G_2_test = pd.DataFrame(X_G_2_test_transformed,columns=X_G_2_test.columns)
    if verbose:
        print("X_G_2_test after scaling: ", X_G_2_test)
    return X_G_2,Y_G_2,X_G_2_train,X_G_2_test,Y_G_2_train,Y_G_2_test

def data_preprocessing_2_10(dataset_2_10,test_size,seed,verbose=False):
    """
    This function is to preprocess the data, removing unsuited samples,
    removing the first column from the dataset of the variables (the first column has the labels of the data),
    and converting into integers the numeric values and scaling the whole dataset.
    @params:
        dataset_2_10 (dataset, pandas or numpy): dataset with same label for 10 year old plantation and remnant forest to work on
        test_size: proportion to split the dataset into training and test set
        seed: random state from which start to split the dataset
        verbose (bool): specify if to print or not the steps and their result
    @returns: dataset divided into X and Y, where X is the dataset with features and samples, and Y the column with the labels of the samples.
    """
    df_2_10 = pd.read_csv(dataset_2_10)
    # Exclusion of samples I2NB, O2NB, O4NA, R1SA, R1SB because bacteria and fungi data with
    # not enough sequencing depth
    # Use boolean indexing to filter out the rows
    # Specify names to remove
    names_to_remove = ['I2NB', 'O2NB', 'O4NA', 'R1SA', 'R1SB']
    df_2_10 = pd.read_csv(dataset_2_10)
    df_2_10 = df_2_10[~df_2_10['Sample_ID'].isin(names_to_remove)]
    if verbose:
        print("df_2_10.shape: ", df_2_10.shape)
        print("df_2_10: ", df_2_10)
    # Now remove the first column, which is "Sample_ID"
    df_2_10 = df_2_10.iloc[:, 1:]
    if verbose: # Print the filtered DataFrame
        print(df_2_10)
        print("df_2_10.shape: ", df_2_10.shape)
    # Python should already see that all columns have numeric data besides the first two
    # Statistical summary of the dataset
    # print(df.describe())
    # Declare feature vector and target variable
    # Now is treated as numpy df
    X_2_10 = df_2_10.iloc[:,1:] # select all rows and all the columns from the third, excluding the first 2 categorical
    if verbose:
        print("X_2_10.shape: ", X_2_10.shape)
        print("X_2_10: ", X_2_10)
    Y_2_10 = df_2_10['Landcover']
    if verbose:
        print("Y_2_10.shape: ", Y_2_10.shape)
        print("Y_2_10: ", Y_2_10)
    # Convert categorical variable into integers ordering them
    map_landcover_label = {"Grassland": 0, "2_10": 1, "24 years old": 2, "Remnant": 3}
    Y_2_10 = [map_landcover_label[lab] for lab in Y_2_10]
    # le = LabelEncoder() # This 2 lines to implement to change into integers the labels without specifications
    # Y_10_R = le.fit_transform(Y_10_R)
    if verbose:
        print("Y_2_10: ", Y_2_10)
    # Datasets separation
    # Split the data into training and test sets
    if verbose:
        print("X_2_10: ", X_2_10)
    X_2_10_train, X_2_10_test, Y_2_10_train, Y_2_10_test = train_test_split(X_2_10, Y_2_10,test_size=test_size,random_state=seed)
    if verbose:
        print("X_2_10_train: ", X_2_10_train)
    # Feature scaling training set
    ms = StandardScaler()
    X_2_10_train_transformed = ms.fit_transform(X_2_10_train)
    X_2_10_train = pd.DataFrame(X_2_10_train_transformed, columns=X_2_10_train.columns)
    if verbose:
        print("X_2_10_train after scaling: ", X_2_10_train)
    # Feature scaling test set
    X_2_10_test_transformed = ms.transform(X_2_10_test)
    X_2_10_test = pd.DataFrame(X_2_10_test_transformed,columns=X_2_10_test.columns)
    if verbose:
        print("X_2_10_test after scaling: ", X_2_10_test)
    return X_2_10,Y_2_10,X_2_10_train,X_2_10_test,Y_2_10_train,Y_2_10_test

def data_preprocessing_G_R(dataset_G_R,test_size,seed,verbose=False):
    """
    This function is to preprocess the data, removing unsuited samples,
    removing the first column from the dataset of the variables (the first column has the labels of the data),
    and converting into integers the numeric values and scaling the whole dataset.
    @params:
        dataset_G_R (dataset, pandas or numpy): dataset with same label for 10 year old plantation and remnant forest to work on
        test_size: proportion to split the dataset into training and test set
        seed: random state from which start to split the dataset
        verbose (bool): specify if to print or not the steps and their result
    @returns: dataset divided into X and Y, where X is the dataset with features and samples, and Y the column with the labels of the samples.
    """
    df_G_R = pd.read_csv(dataset_G_R)
    # Exclusion of samples I2NB, O2NB, O4NA, R1SA, R1SB because bacteria and fungi data with
    # not enough sequencing depth
    # Use boolean indexing to filter out the rows
    # Specify names to remove
    names_to_remove = ['I2NB', 'O2NB', 'O4NA', 'R1SA', 'R1SB']
    df_G_R = pd.read_csv(dataset_G_R)
    df_G_R = df_G_R[~df_G_R['Sample_ID'].isin(names_to_remove)]
    if verbose:
        print("df_G_R.shape: ", df_G_R.shape)
        print("df_G_R: ", df_G_R)
    # Now remove the first column, which is "Sample_ID"
    df_G_R = df_G_R.iloc[:, 1:]
    if verbose: # Print the filtered DataFrame
        print(df_G_R)
        print("df_G_R.shape: ", df_G_R.shape)
    # Python should already see that all columns have numeric data besides the first two
    # Statistical summary of the dataset
    # print(df.describe())
    # Declare feature vector and target variable
    # Now is treated as numpy df
    X_G_R = df_G_R.iloc[:,1:] # select all rows and all the columns from the third, excluding the first 2 categorical
    if verbose:
        print("X_G_R.shape: ", X_G_R.shape)
        print("X_G_R: ", X_G_R)
    Y_G_R = df_G_R['Landcover']
    if verbose:
        print("Y_G_R.shape: ", Y_G_R.shape)
        print("Y_G_R: ", Y_G_R)
    # Convert categorical variable into integers ordering them
    map_landcover_label = {"G_R": 0, "2 years old": 1, "10 years old": 2, "24 years old": 3}
    Y_G_R = [map_landcover_label[lab] for lab in Y_G_R]
    # le = LabelEncoder() # This 2 lines to implement to change into integers the labels without specifications
    # Y_10_R = le.fit_transform(Y_10_R)
    if verbose:
        print("Y_G_R: ", Y_G_R)
    # Datasets separation
    # Split the data into training and test sets
    if verbose:
        print("X_G_R: ", X_G_R)
    X_G_R_train, X_G_R_test, Y_G_R_train, Y_G_R_test = train_test_split(X_G_R, Y_G_R,test_size=test_size,random_state=seed)
    if verbose:
        print("X_G_R_train: ", X_G_R_train)
    # Feature scaling training set
    ms = StandardScaler()
    X_G_R_train_transformed = ms.fit_transform(X_G_R_train)
    X_G_R_train = pd.DataFrame(X_G_R_train_transformed, columns=X_G_R_train.columns)
    if verbose:
        print("X_G_R_train after scaling: ", X_G_R_train)
    # Feature scaling test set
    X_G_R_test_transformed = ms.transform(X_G_R_test)
    X_G_R_test = pd.DataFrame(X_G_R_test_transformed,columns=X_G_R_test.columns)
    if verbose:
        print("X_G_R_test after scaling: ", X_G_R_test)
    return X_G_R,Y_G_R,X_G_R_train,X_G_R_test,Y_G_R_train,Y_G_R_test

def data_preprocessing_2_R(dataset_2_R,test_size,seed,verbose=False):
    """
    This function is to preprocess the data, removing unsuited samples,
    removing the first column from the dataset of the variables (the first column has the labels of the data),
    and converting into integers the numeric values and scaling the whole dataset.
    @params:
        dataset_2_R (dataset, pandas or numpy): dataset with same label for 10 year old plantation and remnant forest to work on
        test_size: proportion to split the dataset into training and test set
        seed: random state from which start to split the dataset
        verbose (bool): specify if to print or not the steps and their result
    @returns: dataset divided into X and Y, where X is the dataset with features and samples, and Y the column with the labels of the samples.
    """
    df_2_R = pd.read_csv(dataset_2_R)
    # Exclusion of samples I2NB, O2NB, O4NA, R1SA, R1SB because bacteria and fungi data with
    # not enough sequencing depth
    # Use boolean indexing to filter out the rows
    # Specify names to remove
    names_to_remove = ['I2NB', 'O2NB', 'O4NA', 'R1SA', 'R1SB']
    df_2_R = pd.read_csv(dataset_2_R)
    df_2_R = df_2_R[~df_2_R['Sample_ID'].isin(names_to_remove)]
    if verbose:
        print("df_2_R.shape: ", df_2_R.shape)
        print("df_2_R: ", df_2_R)
    # Now remove the first column, which is "Sample_ID"
    df_2_R = df_2_R.iloc[:, 1:]
    if verbose: # Print the filtered DataFrame
        print(df_2_R)
        print("df_2_R.shape: ", df_2_R.shape)
    # Python should already see that all columns have numeric data besides the first two
    # Statistical summary of the dataset
    # print(df.describe())
    # Declare feature vector and target variable
    # Now is treated as numpy df
    X_2_R = df_2_R.iloc[:,1:] # select all rows and all the columns from the third, excluding the first 2 categorical
    if verbose:
        print("X_2_R.shape: ", X_2_R.shape)
        print("X_2_R: ", X_2_R)
    Y_2_R = df_2_R['Landcover']
    if verbose:
        print("Y_2_R.shape: ", Y_2_R.shape)
        print("Y_2_R: ", Y_2_R)
    # Convert categorical variable into integers ordering them
    map_landcover_label = {"Grassland": 0, "2_R": 1, "10 years old": 2, "24 years old": 3}
    Y_G_R = [map_landcover_label[lab] for lab in Y_2_R]
    # le = LabelEncoder() # This 2 lines to implement to change into integers the labels without specifications
    # Y_10_R = le.fit_transform(Y_10_R)
    if verbose:
        print("Y_2_R: ", Y_2_R)
    # Datasets separation
    # Split the data into training and test sets
    if verbose:
        print("X_2_R: ", X_2_R)
    X_2_R_train, X_2_R_test, Y_2_R_train, Y_2_R_test = train_test_split(X_2_R, Y_2_R,test_size=test_size,random_state=seed)
    if verbose:
        print("X_2_R_train: ", X_2_R_train)
    # Feature scaling training set
    ms = StandardScaler()
    X_2_R_train_transformed = ms.fit_transform(X_2_R_train)
    X_2_R_train = pd.DataFrame(X_2_R_train_transformed, columns=X_2_R_train.columns)
    if verbose:
        print("X_2_R_train after scaling: ", X_2_R_train)
    # Feature scaling test set
    X_2_R_test_transformed = ms.transform(X_2_R_test)
    X_2_R_test = pd.DataFrame(X_2_R_test_transformed,columns=X_2_R_test.columns)
    if verbose:
        print("X_2_R_test after scaling: ", X_2_R_test)
    return X_2_R,Y_2_R,X_2_R_train,X_2_R_test,Y_2_R_train,Y_2_R_test

if __name__ == '__main__':
    
    # Import dataset using raw string to avoid problems with syntax
    data = r'C:\Users\j_v072\OneDrive - University of the Sunshine Coast (1)\Dottorato cose\PhD\R_Analyses\SER & Holistic approach article\Reviewed_Holistic_Dataset.csv'
    data_10_24 = r'C:\Users\j_v072\OneDrive - University of the Sunshine Coast (1)\Dottorato cose\PhD\R_Analyses\SER & Holistic approach article\Reviewed_Holistic_Dataset_10_24.csv'
    data_24_R = r'C:\Users\j_v072\OneDrive - University of the Sunshine Coast (1)\Dottorato cose\PhD\R_Analyses\SER & Holistic approach article\Reviewed_Holistic_Dataset_24_R.csv'
    data_10_R = r'C:\Users\j_v072\OneDrive - University of the Sunshine Coast (1)\Dottorato cose\PhD\R_Analyses\SER & Holistic approach article\Reviewed_Holistic_Dataset_10_R.csv'
    data_G_2 = r'C:\Users\j_v072\OneDrive - University of the Sunshine Coast (1)\Dottorato cose\PhD\R_Analyses\SER & Holistic approach article\Reviewed_Holistic_Dataset_G_2.csv'
    data_2_10 = r'C:\Users\j_v072\OneDrive - University of the Sunshine Coast (1)\Dottorato cose\PhD\R_Analyses\SER & Holistic approach article\Reviewed_Holistic_Dataset_2_10.csv'
    data_G_R = r'C:\Users\j_v072\OneDrive - University of the Sunshine Coast (1)\Dottorato cose\PhD\R_Analyses\SER & Holistic approach article\Reviewed_Holistic_Dataset_G_R.csv'
    data_2_R = r'C:\Users\j_v072\OneDrive - University of the Sunshine Coast (1)\Dottorato cose\PhD\R_Analyses\SER & Holistic approach article\Reviewed_Holistic_Dataset_2_R.csv'
    
    # Project-related inputs, to set
    parser = argparse.ArgumentParser(description="main")
    parser.add_argument('--random_seed', type=int, default=0, help = 'Random seed to be set.')
    parser.add_argument('--dataset_original', type=str, default=data)
    parser.add_argument('--n_dims_pca', type=int, default=2, help = 'Number of dimensions for dimensionality reduction using PCA.')
    parser.add_argument('--var_threshold',type=int, default=0.10, help = 'Variance threshold to select features.')
    args = parser.parse_args()

    # Set seed to make the code replicabile
    set_seed(random_seed=args.random_seed) 
    # Prepare the datasets
    X, Y, X_train_ss, X_test_ss, Y_train, Y_test, X_train_mm, X_test_mm = data_preprocessing(dataset_path=args.dataset_original, test_size=0.2, seed=args.random_seed, verbose=False)
    # Dimensionality reduction PCA
    #XPCA_train, XPCA_test = pca_dim_reduction(dataset_original=args.dataset_original, data_train=X_train_ss, data_test=X_test_ss, n_dim=args.n_dims_pca,verbose=False, see_plots=False)
    # Dimensionality reduction Variance Threshold
    #XVT_train, XVT_test = variance_tresh_dim_reduction(dataset_training_mm=X_train_mm, dataset_test_mm=X_test_mm, 
    #                                                   variance_threshold=args.var_threshold,seed=args.random_seed, verbose=False)
    # Dimensionality reduction with Pearson correlation
    #XP_train,XP_test=pearson_corr_dim_reduction(Y, X_train_ss, X_test_ss, 0.2, 0.7, seed=args.random_seed, verbose=False, see_plots=False)
    # Cross-validation with dataset without dimensionality reduction
    Decision_tree_crossval (X_train_ss, Y_train,0,data_type=f"no dim reduction, with seed {args.random_seed}", verbose=True)
    Decision_Tree_classification(X_train_ss, Y_train, X_test_ss,Y_test,
                                max_depth=4, min_samples_leaf=2, max_leaf_nodes=5,min_samples_split=2,criterion="gini", random_state=0,
                                data_type=f"no dim reduction, with seed {args.random_seed}",confusion_mat=True, verbose=True, see_plots=True, save_plots=True)
    # Cross-validation with dataset with VT dimensionality reduction
    #Decision_tree_crossval (XVT_train, Y_train,0,data_type=f"VT dim reduction, with seed {args.random_seed}",verbose=True)
    # Decision tree classification with VT dimensionally reduced dataset
    #Decision_Tree_classification(XVT_train, Y_train, XVT_test,Y_test,
    #                             max_depth=3, min_samples_leaf=2, max_leaf_nodes=5,min_samples_split=2,criterion="entropy", random_state=0,
    #                             data_type=f"VT dim reduction, with seed {args.random_seed}",confusion_mat=True, verbose=True, see_plots=False, save_plots=True)
    # Cross-validation with dataset with PCA dimensionality reduction
    #Decision_tree_crossval (XPCA_train, Y_train,0,data_type=f"PCA dim reduction, with seed {args.random_seed}",verbose=True)
    # Decision tree classification with PCA dimensionally reduced dataset
    #Decision_Tree_classification(XPCA_train, Y_train, XPCA_test,Y_test,
    #                              max_depth=3, min_samples_leaf=2, max_leaf_nodes=6,min_samples_split=2,criterion="entropy", random_state=0,
    #                              data_type=f"PCA dim reduction, with seed {args.random_seed}",confusion_mat=True, verbose=True, see_plots=False, save_plots=True)
    # Cross-validation with dataset with P dimensionality reduction
    #Decision_tree_crossval (XP_train, Y_train,0,data_type=f"P dim reduction, with seed {args.random_seed}",verbose=True)
    # Decision tree classification with P dimensionally reduced dataset
    #Decision_Tree_classification(XP_train, Y_train, XP_test,Y_test,
    #                              max_depth=3, min_samples_leaf=2, max_leaf_nodes=7,min_samples_split=2,criterion="gini", random_state=0,
    #                              data_type=f"P dim reduction, with seed {args.random_seed}",confusion_mat=True, verbose=True, see_plots=False, save_plots=True)
    # Classification using merged landcover types, no dim reduction
    # Classification with 10 and 24-year-old plantations together
    #X_10_24,Y_10_24,X_10_24_train,X_10_24_test,Y_10_24_train,Y_10_24_test = data_preprocessing_10_24(dataset_10_24=data_10_24, verbose=False,test_size=0.2,seed=args.random_seed)
    # Cross-validation
    #Decision_tree_crossval (X_10_24_train,Y_10_24_train,0,data_type=f"no dim reduction, with seed {args.random_seed}, 10-24 together",verbose=True)
    # Classification
    #Decision_Tree_classification(X_10_24_train, Y_10_24_train, X_10_24_test,Y_10_24_test,
    #                              max_depth=2, min_samples_leaf=2, max_leaf_nodes=4,min_samples_split=2,criterion="entropy", random_state=0,
    #                              data_type=f"no dim reduction, with seed {args.random_seed}, 10-24 together",confusion_mat=True, verbose=True, see_plots=True, save_plots=True,accuracy_per_class=True)
    # Classification with 24-year-old plantation and remnant forest together  
    #X_24_R,Y_24_R,X_24_R_train,X_24_R_test,Y_24_R_train,Y_24_R_test = data_preprocessing_24_R(dataset_24_R=data_24_R, verbose=False,test_size=0.2,seed=args.random_seed)
    # Cross-validation
    #Decision_tree_crossval (X_24_R_train,Y_24_R_train,0,data_type=f"no dim reduction, with seed {args.random_seed}, 24-R together",verbose=False)
    # Classification
    #Decision_Tree_classification(X_24_R_train, Y_24_R_train, X_24_R_test,Y_24_R_test,
    #                              max_depth=2, min_samples_leaf=2, max_leaf_nodes=4,min_samples_split=2,criterion="entropy", random_state=0,
    #                              data_type=f"no dim reduction, with seed {args.random_seed}, 24-R together",confusion_mat=True, verbose=True, see_plots=False, save_plots=True,accuracy_per_class=True)
    # Classification with 10-year-old plantation and remnant forest together
    #X_10_R,Y_10_R,X_10_R_train,X_10_R_test,Y_10_R_train,Y_10_R_test = data_preprocessing_10_R(dataset_10_R=data_10_R, verbose=False,test_size=0.2,seed=args.random_seed)
    # Cross-validation
    #Decision_tree_crossval (X_10_R_train,Y_10_R_train,0,data_type=f"no dim reduction, with seed {args.random_seed}, 10-R together", verbose=False)
    #Decision_Tree_classification(X_10_R_train, Y_10_R_train, X_10_R_test,Y_10_R_test,
    #                             max_depth=3, min_samples_leaf=2, max_leaf_nodes=6,min_samples_split=2,criterion="entropy", random_state=0,
    #                             data_type=f"no dim reduction, with seed {args.random_seed}, 10-R together",confusion_mat=False, verbose=False, see_plots=True, save_plots=True,accuracy_per_class=True)

    # Classification with grassland and 2-year-old together
    #X_G_2,Y_G_2,X_G_2_train,X_G_2_test,Y_G_2_train,Y_G_2_test = data_preprocessing_G_2(dataset_G_2=data_G_2, verbose=False,test_size=0.2,seed=args.random_seed)
    # Cross-validation
    #Decision_tree_crossval (X_G_2_train,Y_G_2_train,0,data_type=f"no dim reduction, with seed {args.random_seed}, G-2 together", verbose=True)
    #Decision_Tree_classification(X_G_2_train, Y_G_2_train, X_G_2_test,Y_G_2_test,
    #                             max_depth=3, min_samples_leaf=2, max_leaf_nodes=4,min_samples_split=2,criterion="entropy", random_state=0,
    #                             data_type=f"no dim reduction, with seed {args.random_seed}, G-2 together",confusion_mat=False, verbose=False, see_plots=False, save_plots=True,accuracy_per_class=True)
    
    # Classification with 2- and 10-year-old together
    #X_2_10,Y_2_10,X_2_10_train,X_2_10_test,Y_2_10_train,Y_2_10_test = data_preprocessing_2_10(dataset_2_10=data_2_10, verbose=False,test_size=0.2,seed=args.random_seed)
    # Cross-validation
    #Decision_tree_crossval (X_2_10_train,Y_2_10_train,0,data_type=f"no dim reduction, with seed {args.random_seed}, 2-10 together", verbose=True)
    #Decision_Tree_classification(X_2_10_train, Y_2_10_train, X_2_10_test,Y_2_10_test,
    #                             max_depth=2, min_samples_leaf=2, max_leaf_nodes=4,min_samples_split=2,criterion="entropy", random_state=0,
    #                             data_type=f"no dim reduction, with seed {args.random_seed}, 2-10 together",confusion_mat=False, verbose=False, see_plots=False, save_plots=True,accuracy_per_class=True)
    
    # Classification with grassland and Remnant forest together
    #X_G_R,Y_G_R,X_G_R_train,X_G_R_test,Y_G_R_train,Y_G_R_test = data_preprocessing_G_R(dataset_G_R=data_G_R, verbose=False,test_size=0.2,seed=args.random_seed)
    # Cross-validation
    #Decision_tree_crossval (X_G_R_train,Y_G_R_train,0,data_type=f"no dim reduction, with seed {args.random_seed}, G-R together", verbose=True)
    #Decision_Tree_classification(X_G_R_train, Y_G_R_train, X_G_R_test,Y_G_R_test,
    #                             max_depth=4, min_samples_leaf=4, max_leaf_nodes=5,min_samples_split=2,criterion="entropy", random_state=0,
    #                             data_type=f"no dim reduction, with seed {args.random_seed}, G-R together",confusion_mat=False, verbose=False, see_plots=False, save_plots=True,accuracy_per_class=True)
    
    # Classification with 2 years old plantation and Remnant forest together
    #X_2_R,Y_2_R,X_2_R_train,X_2_R_test,Y_2_R_train,Y_2_R_test = data_preprocessing_2_R(dataset_2_R=data_2_R, verbose=False,test_size=0.2,seed=args.random_seed)
    # Cross-validation
    #Decision_tree_crossval (X_2_R_train,Y_2_R_train,0,data_type=f"no dim reduction, with seed {args.random_seed}, 2-R together", verbose=True)
    #Decision_Tree_classification(X_2_R_train, Y_2_R_train, X_2_R_test,Y_2_R_test,
    #                             max_depth=3, min_samples_leaf=4, max_leaf_nodes=4,min_samples_split=2,criterion="entropy", random_state=0,
    #                             data_type=f"no dim reduction, with seed {args.random_seed}, 2-R together",confusion_mat=False, verbose=False, see_plots=False, save_plots=True,accuracy_per_class=True)