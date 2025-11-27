# Supervised and Unsupervised ML algorithms to understand the restoration potential of Acacia mangium
# KNN analysis

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

def pearson_corr_dim_reduction (labels, data_train, data_test, corr_threshold_with_target,corr_threshold_between_feat,seed, verbose=False, see_plots=True):
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
    # Select the variables based on choosen threshold
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

def knn_crossvalidation(data_train, labels_train, data_type, Choose_k=True, see_plots=False, save_plots=True):
    '''
    Function to perform KNN classification with cross validation. By default the cross validation
    is 5 fold.
    @params
        data_train: dataset on which the training of the KNN model is performed
        labels_train: list of labels related to the dataset of training
        data_type: type of dataset, if with or without dimensionality reduction
        verbose (bool): to print results
        confusion_matrix (bool): to print the confusion matrix
        see_plots (bool): to see plots
        save_plots (bool): to save the generated plots
    '''
    # set the range of k-values to try. Here 1 to 10
    k_values = [i for i in range (1,11)]
    scores = []
    # Run the algorithm considering a fivefold cross-validation
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, data_train, labels_train, cv=5) #Cross-validation just with training set
        scores.append(np.mean(score))
    if see_plots:
        # Plot the accuracy variation according to the consideration of a different number of k
        sns.lineplot(x = k_values, y = scores, marker = 'o')
        plt.title(f"5 fold Cross validation with {data_type}")
        plt.xlabel("K Values")
        plt.ylabel(f"Accuracy Score cross-validation with {data_type}")
        plt.show()
    if save_plots:
        os.makedirs('KNN_plots', exist_ok=True)
        # Plot the accuracy variation according to the consideration of a different number of k
        sns.lineplot(x = k_values, y = scores, marker = 'o')
        plt.title(f"5 fold Cross validation with {data_type}")
        plt.xlabel("K Values")
        plt.ylabel(f"Accuracy Score cross-validation with {data_type}")
        plt.savefig(f"KNN_plots/KNN_Accuracy_crossval_{data_type}.png")
    if Choose_k:
        # Print k values and corresponding accuracy score
        for k, accuracy in zip(k_values, scores):
            print(f"k = {k}, Accuracy cross-val. with {data_type} = {accuracy}")

def knn_classification(data_train, labels_train, data_test, labels_test, best_k, data_type, verbose=False, confusion_matrix=True, see_plots=True, save_plots=True): 
    '''
    Function to perform KNN classification.
    @params
        data_train: dataset on which the training of the KNN model is performed
        labels_train: list of labels related to the dataset of training
        data_test: dataset on which the classification with the KNN model is performed
        labels_test: list of labels related to the dataset of test
        best_k: value of neighbor which gives the best accuracy
        data_type: type of dataset, if with or without dimensionality reduction
        verbose (bool): to print results
        confusion_matrix (bool): to print the confusion matrix
        see_plots: To display or not the plots
        save_plots (bool): to save the generated plots
    '''
    # KNN algorithm run with the identified best k-value and the splitted dataset (see region "General dataset preparation" above)
    # Training phase
    # KNN algorithm training
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn = knn.fit(data_train, labels_train)
    # KNN algorithm test
    Y_pred = knn.predict(data_test)
    if verbose:
        # Model Accuracy: how often is the classifier correct?
        print(f"Accuracy with {data_type}:",metrics.accuracy_score(labels_test, Y_pred))
        # Model Precision
        # With one of these:
        print(f"Precision with {data_type} (micro):", metrics.precision_score(labels_test, Y_pred, average='micro'))
        print(f"Precision with {data_type} (macro):", metrics.precision_score(labels_test, Y_pred, average='macro'))
        print(f"Precision with {data_type} (weighted):", metrics.precision_score(labels_test, Y_pred, average='weighted'))
        print(f"Precision with {data_type} (per class):", metrics.precision_score(labels_test, Y_pred, average=None))
        # Model Recall
        # With one of these:
        print(f"Recall with {data_type}(micro):",metrics.recall_score(labels_test, Y_pred, average='micro'))
        print(f"Recall with {data_type}(macro):",metrics.recall_score(labels_test, Y_pred, average='macro'))
        print(f"Recall with {data_type}(weighted):",metrics.recall_score(labels_test, Y_pred, average='weighted'))
        print(f"Recall with {data_type}(per class):",metrics.recall_score(labels_test, Y_pred, average=None))
    if see_plots:
        # Scatter plot
        if isinstance(data_test, np.ndarray): #Change into pandas dataframe if numpy
            data_test = pd.DataFrame(data_test)
        plt.figure()
        plt.scatter(data_test.loc[:, (data_test.columns[0])], data_test.loc[:, (data_test.columns[1])], c=Y_pred, cmap='viridis', s=50)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'KNN with {data_type}')
        plt.show()
    if save_plots:
        os.makedirs('KNN_plots', exist_ok=True)
        # Scatter plot
        if isinstance(data_test, np.ndarray): #Change into pandas dataframe if numpy
            data_test = pd.DataFrame(data_test)
        plt.figure()
        plt.scatter(data_test.loc[:, (data_test.columns[0])], data_test.loc[:, (data_test.columns[1])], c=Y_pred, cmap='viridis', s=50)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'KNN with {data_type}')
        plt.savefig(f"KNN_plots/KNN_with_{data_type}.png")
    if see_plots:
        # Visualize the results with data labelled according to their landcover type
        # Scatter plot
        if isinstance(data_test, np.ndarray): #Change into pandas dataframe if numpy
            data_test = pd.DataFrame(data_test)
        # Define discrete colors
        colors = ['red', 'pink', 'lightblue', 'orange', 'green']
        # Create a ListedColormap for plot function
        cmap = ListedColormap(colors)
        plt.figure(figsize=(10, 6))
        plt.scatter(data_test.loc[:, (data_test.columns[0])], data_test.loc[:, (data_test.columns[1])],c=labels_test, cmap=cmap)
        plt.title(f'KNN Results with samples colored according to their landcover type and {data_type}')
        # Create a manual legend
        unique_labels = np.unique(Y_test)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, label=f'{label}') for i, label in enumerate(unique_labels)]
        plt.legend(handles=legend_elements)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'KNN with {data_type} and samples colored according to their landcover type')
        plt.show()
    if save_plots:
        os.makedirs('KNN_plots', exist_ok=True)
        # Visualize the results with data labelled according to their landcover type
        # Scatter plot
        if isinstance(data_test, np.ndarray): #Change into pandas dataframe if numpy
            data_test = pd.DataFrame(data_test)
        # Define discrete colors
        colors = ['red', 'pink', 'lightblue', 'orange', 'green']
        # Create a ListedColormap for plot function
        cmap = ListedColormap(colors)
        plt.figure(figsize=(10, 6))
        plt.scatter(data_test.loc[:, (data_test.columns[0])], data_test.loc[:, (data_test.columns[1])],c=labels_test, cmap=cmap)
        plt.title(f'KNN Results with samples colored according to their landcover type and {data_type}')
        # Create a manual legend
        unique_labels = np.unique(Y_test)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, label=f'{label}') for i, label in enumerate(unique_labels)]
        plt.legend(handles=legend_elements)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'KNN with {data_type} and samples colored according to their landcover type')
        plt.savefig(f"KNN_plots/KNN_with_{data_type}_landcover_labels.png")
    if confusion_matrix:
        # Confusion matrix
        # Display confusion matrix
        cross_tab = pd.crosstab(labels_test, Y_pred)
        print(f"confusion matrix KNN  with {data_type}")
        print(cross_tab)

if __name__ == '__main__':
    
    # Import dataset using raw string to avoid problems with syntax
    data = r'C:\Users\j_v072\OneDrive - University of the Sunshine Coast (1)\Dottorato cose\PhD\R_Analyses\SER & Holistic approach article\Reviewed_Holistic_Dataset.csv'

    # Project-related inputs, to set
    parser = argparse.ArgumentParser(description="main")
    parser.add_argument('--random_seed', type=int, default=3, help = 'Random seed to be set.')
    parser.add_argument('--dataset_original', type=str, default=data)
    parser.add_argument('--n_dims_pca', type=int, default=2, help = 'Number of dimensions for dimensionality reduction using PCA.')
    parser.add_argument('--var_threshold',type=int, default=0.10, help = 'Variance threshold to select features.')
    args = parser.parse_args()

    #random_seed = 3 #run with 0,1,2,3
    # Set seed to make the code replicabile
    set_seed(random_seed=args.random_seed) 
    # Prepare the datasets
    X, Y, X_train_ss, X_test_ss, Y_train, Y_test, X_train_mm, X_test_mm = data_preprocessing(dataset_path=args.dataset_original, test_size=0.2, seed=args.random_seed, verbose=False)
    # Dimensionality reduction PCA
    XPCA_train, XPCA_test = pca_dim_reduction(dataset_original=args.dataset_original, data_train=X_train_ss, data_test=X_test_ss, n_dim=args.n_dims_pca, see_plots=False, verbose=True)
    # Dimensionality reduction Variance Threshold
    XVT_train, XVT_test = variance_tresh_dim_reduction(dataset_training_mm=X_train_mm, dataset_test_mm=X_test_mm, 
                                                       variance_threshold=args.var_threshold,seed=args.random_seed, verbose=True)
    # Dimensionality reduction with Pearson correlation
    XP_train,XP_test=pearson_corr_dim_reduction(Y, X_train_ss, X_test_ss, 0.2, 0.7, seed=args.random_seed, verbose=True, see_plots=False)
    # Cross-validation without dimensionality reduction
    knn_crossvalidation(X_train_ss,Y_train,data_type=f'no dim reduction, seed {args.random_seed}',Choose_k=False,see_plots=False, save_plots=True)
    # KNN without dimensionality reduction
    knn_classification(X_train_ss,Y_train,X_test_ss,Y_test,best_k=1,data_type=f'no dim reduction, seed {args.random_seed}',see_plots=False, save_plots=True, verbose=True)
    # Cross-validation with VT dimensionality args.random_seed
    knn_crossvalidation(XVT_train,Y_train,data_type=f'VT dim reduction, seed {args.random_seed}',Choose_k=False,see_plots=False, save_plots=True)
    # KNN with VT dimensionality reduction
    knn_classification(XVT_train,Y_train,XVT_test,Y_test,best_k=4,data_type=f'VT dim reduction, seed {args.random_seed}',see_plots=False, save_plots=True, verbose=True)
    # Cross-validation with PCA dimensionality reduction
    knn_crossvalidation(XPCA_train,Y_train,data_type=f'PCA dim reduction, seed {args.random_seed}',Choose_k=False,see_plots=False, save_plots=True)
    # KNN with PCA dimensionality reduction
    knn_classification(XPCA_train,Y_train,XPCA_test,Y_test,best_k=3,data_type=f'PCA dim reduction, seed {args.random_seed}', verbose=True,see_plots=False, save_plots=True)
    # Cross-validation with P dimensionality reduction
    knn_crossvalidation(XP_train,Y_train,data_type=f'P dim reduction, seed {args.random_seed}',Choose_k=False,see_plots=False, save_plots=True)
    # KNN with P dimensionality reduction
    knn_classification(XP_train,Y_train,XP_test,Y_test,best_k=8,data_type=f'P dim reduction, seed {args.random_seed}', verbose=True,see_plots=False, save_plots=True)