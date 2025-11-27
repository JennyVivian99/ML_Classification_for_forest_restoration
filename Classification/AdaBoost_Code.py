# Supervised and Unsupervised ML algorithms to understand the restoration potential of Acacia mangium
# AdaBoost analysis

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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Don't import the libraries of methods that you are not using
#endregion

def set_seed(random_seed=0):
    np.random.seed(random_seed)

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
    # Select the variables based on chosed threshold
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
                columns_to_keep[j] = False
    if verbose:
        print("Columns to keep: ", columns_to_keep)
    selected_features_pairwise = regression_variables.columns[columns_to_keep]
    if verbose:
        print("\nFeatures remaining after removing highly correlated features (threshold >= {}):\n".format(corr_threshold_between_feat), selected_features_pairwise)
    # Filter the columns
    XP_train = XP_train[selected_features_pairwise]
    XP_test = data_test[selected_features_pairwise]
    if verbose:
        print("XP_train: ", XP_train.shape)
        print("XP_test: ", XP_test.shape)
    return XP_train, XP_test

def AdaBoost_crossval(dataset_training, labels_training, kernel, C, gamma, max_depth, min_samples_leaf, max_leaf_nodes, min_samples_split, criterion, random_state, data_type, verbose=True):
    """
    This is used to train the AdaBoost algorithm considering different hyperparameters. Specifically
    different base learners, number of estimators, and learning rate.
    Then, the output is printed and saved to understand the combination yelding the highest accuracy.
    @params:
        dataset_training: training dataset
        labels_training: list of labels of the training dataset
        TO FINISH
        data_type (bool): type of dimensionality reduction (if done)
        verbose (bool): to print or not the results
    @returns: decision_tree_base, svm_base
    """
    # Run the algorithm considering decision tree, and svm, with different
    # n_estimators and learning rates
    # SVM used is the best performing one from SVM analysis for each type of dataset: rbf kernel, C= 10 and gamma: 0.001
    # DT used is the best performing and simplest one from DT analysis, thus the one from the original dataset
    # analysis. The one from adopted for the VT dataset also yelded 1 as accuracy, but adopted a different criterion for splitting. Therefore,
    # we choose the first one, being different from the VT just for max depth of 4 instead of 3 and the default criterion (gini).
    # The other common parameters are: maximum depth of 4, a minimum number of samples per leaf and for splitting of 2, and a maximum number of leaves of 5.
    svm_base=svm.SVC(kernel=kernel, C=C, gamma=gamma)
    decision_tree_base=DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,min_samples_split=min_samples_split,criterion=criterion, random_state=random_state)
    estimators=[svm_base, decision_tree_base]
    num_estimators=[10,20,30,40,50,60,70,80]
    learning_rates=[0.001,0.01,0.1,1,10,100]
    results = {}  # Dictionary to store the results
    for c in estimators:
        for k in num_estimators:
            for l in learning_rates:
                    # Run the classifier
                    AdaBoostClass = AdaBoostClassifier(estimator=c,n_estimators=k,learning_rate=l)
                    # Calculate and store the accuracy scores from 5fold cross-validation on the training dataset
                    scores = cross_val_score(AdaBoostClass, dataset_training, labels_training, cv=5) #Cross-validation just with training set
                    # Store the mean cross-validation score in the results dictionary
                    results[f"Base learner:{c.__class__.__name__}, n^ estimators:{k}, learning rate:{l}"] = np.mean(scores)
    # Visualise the results
    if verbose:
        for key, value in results.items():
             print(f"Algorithm {key}, mean accuracy 5 fold-cv: {value:.4f}")
    # Save to Excel file to understand which is the best base learner:
    results_df = pd.DataFrame(list(results.items()), columns=['Hyperparameters', 'Test Accuracy'])
    results_df.to_csv(f"AdaBoostTrainingOutput_{data_type}.csv", index=False)
    return decision_tree_base, svm_base

def AdaBoost_classification(dataset_training, labels_training,dataset_test,labels_test,estimator,n_estimators,learning_rate, data_type, verbose=True, see_plots=True,save_plots=True,confusion_mat=True,accuracy_per_class=False):
    '''
    This function implements the classification through AdaBoost classifier, from scikit learn. It also provides
    the metrics, plots, and confusion matrix of the result.
    @params:
        dataset_training: training dataset
        labels_training: list of labels of the training dataset
        dataset_test: test dataset
        labels_test: list of labels of the test dataset
        estimator: type of estimator to implement (base learner)
        n_estimators: number of estimators (base learners) to implement
        learning_rate: rate of the algorithm's learning (from each iteration)
        data_type (bool): type of dimensionality reduction (if done)
        verbose (bool): to print or not the results
        see_plots (bool): to display or not the plots
        save_plots (bool): to save or not the plots (regardless see_plots indication)
        confusion_mat (bool): to display or not the confusion matrix
    '''
    # Create AdaBoost classifer object
    AdaBoostClass = AdaBoostClassifier(estimator=estimator,n_estimators=n_estimators,learning_rate=learning_rate)
    # Train AdaBoost Classifer
    model = AdaBoostClass.fit(dataset_training, labels_training)
    #Predict the response for test dataset
    Y_pred = model.predict(dataset_test)
    if see_plots:
        # Scatter plot
        if isinstance(dataset_test, np.ndarray): #Change into pandas dataframe if numpy
            dataset_test = pd.DataFrame(dataset_test)
        plt.figure()
        plt.scatter(dataset_test.loc[:, (dataset_test.columns[0])], dataset_test.loc[:, (dataset_test.columns[1])], c=Y_pred, cmap='viridis', s=50)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'KNN with {data_type}')
        plt.show()
    if save_plots:
        os.makedirs('AdaBoost_plots', exist_ok=True)
        # Scatter plot
        if isinstance(dataset_test, np.ndarray): #Change into pandas dataframe if numpy
            dataset_test = pd.DataFrame(dataset_test)
        plt.figure()
        plt.scatter(dataset_test.loc[:, (dataset_test.columns[0])], dataset_test.loc[:, (dataset_test.columns[1])], c=Y_pred, cmap='viridis', s=50)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'AdaBoost with {data_type}')
        plt.savefig(f"AdaBoost_plots/AdaBoost_with_{data_type}.png")
    if see_plots:
        # Visualize the results with data labelled according to their landcover type
        # Scatter plot
        if isinstance(dataset_test, np.ndarray): #Change into pandas dataframe if numpy
            dataset_test = pd.DataFrame(dataset_test)
        # Define discrete colors
        colors = ['red', 'pink', 'lightblue', 'orange', 'green']
        # Create a ListedColormap for plot function
        cmap = ListedColormap(colors)
        plt.figure(figsize=(10, 6))
        plt.scatter(dataset_test.loc[:, (dataset_test.columns[0])], dataset_test.loc[:, (dataset_test.columns[1])],c=labels_test, cmap=cmap)
        plt.title(f'AdaBoost Results with samples colored according to their landcover type and {data_type}')
        # Create a manual legend
        unique_labels = np.unique(labels_test)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, label=f'{label}') for i, label in enumerate(unique_labels)]
        plt.legend(handles=legend_elements)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'AdaBoost with {data_type} and samples colored according to their landcover type')
        plt.show()
    if save_plots:
        os.makedirs('AdaBoost_plots', exist_ok=True)
        # Visualize the results with data labelled according to their landcover type
        # Scatter plot
        if isinstance(dataset_test, np.ndarray): #Change into pandas dataframe if numpy
            dataset_test = pd.DataFrame(dataset_test)
        # Define discrete colors
        colors = ['red', 'pink', 'lightblue', 'orange', 'green']
        # Create a ListedColormap for plot function
        cmap = ListedColormap(colors)
        plt.figure(figsize=(10, 6))
        plt.scatter(dataset_test.loc[:, (dataset_test.columns[0])], dataset_test.loc[:, (dataset_test.columns[1])],c=labels_test, cmap=cmap)
        plt.title(f'AdaBoost Results with samples colored according to their landcover type and {data_type}')
        # Create a manual legend
        unique_labels = np.unique(labels_test)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10, label=f'{label}') for i, label in enumerate(unique_labels)]
        plt.legend(handles=legend_elements)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'AdaBoost with {data_type} and samples colored according to their landcover type')
        plt.savefig(f"AdaBoost_plots/AdaBoost_with_{data_type}_landcover_labels.png")
    # Model Accuracy
    if verbose:
        print(f"Accuracy with {data_type}:",metrics.accuracy_score(labels_test, Y_pred))
        # Model Precision
        # With one of these:
        print(f"Precision with {data_type} (micro):", metrics.precision_score(labels_test, Y_pred, average='micro'))
        print(f"Precision with {data_type} (macro):", metrics.precision_score(labels_test, Y_pred, average='macro'))
        print(f"Precision with {data_type} (weighted):", metrics.precision_score(labels_test, Y_pred, average='weighted'))
        print(f"Precision with {data_type} (per class):", metrics.precision_score(labels_test, Y_pred, average=None))
        # Model Recall
        # With one of these:
        print(f"Recall with {data_type} (micro):",metrics.recall_score(labels_test, Y_pred, average='micro'))
        print(f"Recall with {data_type} (macro):",metrics.recall_score(labels_test, Y_pred, average='macro'))
        print(f"Recall with {data_type} (weighted):",metrics.recall_score(labels_test, Y_pred, average='weighted'))
        print(f"Recall with {data_type} (per class):",metrics.recall_score(labels_test, Y_pred, average=None))
    if accuracy_per_class:
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
        print(f"Accuracy table AdaBoost  with {data_type}",Acc_per_class)
    if confusion_mat:
        # Display confusion_mat
        cross_tab = pd.crosstab(labels_test, Y_pred)
        print(f"confusion matrix AdaBoost  with {data_type}")
        print(cross_tab)

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
    Y_2_R = [map_landcover_label[lab] for lab in Y_2_R]
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
    parser.add_argument('--n_dims_pca', type=int, default=3, help = 'Number of dimensions for dimensionality reduction using PCA.')
    parser.add_argument('--var_threshold',type=int, default=0.10, help = 'Variance threshold to select features.')
    args = parser.parse_args()

    # Set seed to make the code replicabile
    set_seed(random_seed=args.random_seed) 
    # Prepare the datasets
    #X, Y, X_train_ss, X_test_ss, Y_train, Y_test, X_train_mm, X_test_mm = data_preprocessing(dataset_path=args.dataset_original, test_size=0.2, seed=args.random_seed, verbose=False)
    # Dimensionality reduction PCA
    #XPCA_train, XPCA_test = pca_dim_reduction(dataset_original=args.dataset_original, data_train=X_train_ss, data_test=X_test_ss, n_dim=args.n_dims_pca, see_plots=False)
    # Dimensionality reduction Variance Threshold
    #XVT_train, XVT_test = variance_tresh_dim_reduction(dataset_training_mm=X_train_mm, dataset_test_mm=X_test_mm, 
    #                                                   variance_threshold=args.var_threshold, verbose=False)
    # Dimensionality reduction with Pearson correlation
    #XP_train,XP_test=pearson_corr_dim_reduction(Y, X_train_ss, X_test_ss, 0.2, 0.7, seed=args.random_seed,verbose=True, see_plots=False)
    # AdaBoost crossvalidation without dimensionality reduction
    #decision_tree_base, svm_base=AdaBoost_crossval(dataset_training=X_train_ss, labels_training=Y_train,kernel="rbf",C=10,gamma=0.001, max_depth=4, min_samples_leaf=2, max_leaf_nodes=5,min_samples_split=2,criterion="gini", random_state=0,data_type=f"no dim red, with seed {args.random_seed}",verbose=False)
    # AdaBoost classification
    #AdaBoost_classification(dataset_training=X_train_ss, labels_training=Y_train, dataset_test=X_test_ss,labels_test=Y_test,estimator= decision_tree_base,n_estimators=10,learning_rate=0.001, data_type=f"no dim reduction, with seed {args.random_seed}", verbose=True,see_plots=False,save_plots=True,confusion_mat=True)
    # AdaBoost crossvalidation with VT dimensionality reduction
    #decision_tree_base, svm_base=AdaBoost_crossval(dataset_training=XVT_train, labels_training=Y_train, kernel="rbf",C=10,gamma=1,max_depth=3, min_samples_leaf=2, max_leaf_nodes=5,min_samples_split=2,criterion="entropy", random_state=0,data_type=f"VT dim red, with seed {args.random_seed}",verbose=False)
    # AdaBoost classification
    #AdaBoost_classification(dataset_training=XVT_train, labels_training=Y_train, dataset_test=XVT_test,labels_test=Y_test,estimator= decision_tree_base,n_estimators=10,learning_rate=0.001, data_type=f"VT dim reduction, with seed {args.random_seed}", verbose=True,see_plots=False,save_plots=True,confusion_mat=True)
    # AdaBoost crossvalidation with P dimensionality reduction
    #decision_tree_base, svm_base=AdaBoost_crossval(dataset_training=XP_train, labels_training=Y_train, kernel="rbf",C=10,gamma=0.001, max_depth=3, min_samples_leaf=2, max_leaf_nodes=7,min_samples_split=2,criterion="gini", random_state=0,data_type=f"P dim red, with seed {args.random_seed}",verbose=False)
    # AdaBoost classification
    #AdaBoost_classification(dataset_training=XP_train, labels_training=Y_train, dataset_test=XP_test,labels_test=Y_test,estimator= decision_tree_base,n_estimators=20,learning_rate=10, data_type=f"P dim reduction, with seed {args.random_seed}", verbose=True,see_plots=False,save_plots=True,confusion_mat=True)
    # AdaBoost crossvalidation with PCA dimensionality reduction
    #decision_tree_base, svm_base=AdaBoost_crossval(dataset_training=XPCA_train, labels_training=Y_train, kernel="rbf",C=10,gamma=0.001,max_depth=3, min_samples_leaf=2, max_leaf_nodes=6,min_samples_split=2,criterion="entropy", random_state=0, data_type=f"PCA dim red, with seed {args.random_seed}",verbose=True)
    # AdaBoost classification
    #AdaBoost_classification(dataset_training=XPCA_train, labels_training=Y_train, dataset_test=XPCA_test,labels_test=Y_test,estimator= decision_tree_base,n_estimators=10,learning_rate=0.01, data_type=f"PCA dim red, with seed {args.random_seed}", verbose=True,see_plots=False,save_plots=True,confusion_mat=True)
    
    # AdaBoost classification without dimensionality reduction and considering 10-24-year-old plantations and remnant forest
    # Classification with 10 and 24-year-old plantations together, using best parameters for original dataset of SVMs and DT
    #X_10_24,Y_10_24,X_10_24_train,X_10_24_test,Y_10_24_train,Y_10_24_test = data_preprocessing_10_24(dataset_10_24=data_10_24, verbose=False,test_size=0.2,seed=args.random_seed)
    #decision_tree_base, svm_base=AdaBoost_crossval(dataset_training=X_10_24_train, labels_training=Y_10_24_train,kernel="rbf",C=10,gamma=0.001, max_depth=4, min_samples_leaf=2, max_leaf_nodes=5,min_samples_split=2,criterion="gini", random_state=0,data_type=f"no dim red 10_24 together, with seed {args.random_seed}",verbose=True)
    # AdaBoost classification
    #AdaBoost_classification(dataset_training=X_10_24_train, labels_training=Y_10_24_train, dataset_test=X_10_24_test,labels_test=Y_10_24_test,estimator= decision_tree_base,n_estimators=10,learning_rate=0.01, data_type=f"no dim red 10_24 together, with seed {args.random_seed}", verbose=True,see_plots=False,save_plots=True,confusion_mat=True,accuracy_per_class=True)
    # Classification with 24-year-old plantation and remnant forest together, with best SVMs for original dataset and DT for original dataset
    #X_24_R,Y_24_R,X_24_R_train,X_24_R_test,Y_24_R_train,Y_24_R_test = data_preprocessing_24_R(dataset_24_R=data_24_R, verbose=False,test_size=0.2,seed=args.random_seed)
    #decision_tree_base, svm_base=AdaBoost_crossval(dataset_training=X_24_R_train, labels_training=Y_24_R_train,kernel="rbf",C=10,gamma=0.001, max_depth=4, min_samples_leaf=2, max_leaf_nodes=5,min_samples_split=2,criterion="gini", random_state=0,data_type=f"no dim red 24_R together, with seed {args.random_seed}",verbose=True)
    # AdaBoost classification
    #AdaBoost_classification (dataset_training=X_24_R_train, labels_training=Y_24_R_train, dataset_test=X_24_R_test,labels_test=Y_24_R_test,estimator= decision_tree_base,n_estimators=10,learning_rate=0.01, data_type=f"no dim red 24_R together, with seed {args.random_seed}", verbose=True,see_plots=False,save_plots=True,confusion_mat=True,accuracy_per_class=True)
    # Classification with 10-year-old plantation and remnant forest together, with best SVMs for original dataset and DT for original dataset
    #X_10_R,Y_10_R,X_10_R_train,X_10_R_test,Y_10_R_train,Y_10_R_test = data_preprocessing_10_R(dataset_10_R=data_10_R, verbose=False,test_size=0.2,seed=args.random_seed)
    #decision_tree_base, svm_base=AdaBoost_crossval(dataset_training=X_10_R_train, labels_training=Y_10_R_train,kernel="rbf",C=10,gamma=0.001,max_depth=4, min_samples_leaf=2, max_leaf_nodes=5,min_samples_split=2,criterion="gini", random_state=0, data_type=f"no dim red 10_R together, with seed {args.random_seed}",verbose=False)
    # AdaBoost classification
    #AdaBoost_classification(dataset_training=X_10_R_train, labels_training=Y_10_R_train, dataset_test=X_10_R_test,labels_test=Y_10_R_test,estimator= decision_tree_base,n_estimators=40,learning_rate=1, data_type=f"no dim red 10_R together, with seed {args.random_seed}", verbose=False,see_plots=False,save_plots=True,confusion_mat=False,accuracy_per_class=True)
    
    # Classification with grassland and 2 years old plantation together, with best SVMs for original dataset and DT for original dataset
    #X_G_2,Y_G_2,X_G_2_train,X_G_2_test,Y_G_2_train,Y_G_2_test = data_preprocessing_G_2(dataset_G_2=data_G_2, verbose=False,test_size=0.2,seed=args.random_seed)
    #decision_tree_base, svm_base=AdaBoost_crossval(dataset_training=X_G_2_train, labels_training=Y_G_2_train,kernel="rbf",C=10,gamma=0.001,max_depth=4, min_samples_leaf=2, max_leaf_nodes=5,min_samples_split=2,criterion="gini", random_state=0, data_type=f"no dim red G_2 together, with seed {args.random_seed}",verbose=False)
    # AdaBoost classification
    #AdaBoost_classification(dataset_training=X_G_2_train, labels_training=Y_G_2_train, dataset_test=X_G_2_test,labels_test=Y_G_2_test,estimator= decision_tree_base,n_estimators=10,learning_rate=0.01, data_type=f"no dim red G_2 together, with seed {args.random_seed}", verbose=False,see_plots=False,save_plots=True,confusion_mat=False,accuracy_per_class=True)
    # Classification with 2 years old and 10 years old plantations together, with best SVMs for original dataset and DT for original dataset
    #X_2_10,Y_2_10,X_2_10_train,X_2_10_test,Y_2_10_train,Y_2_10_test = data_preprocessing_2_10(dataset_2_10=data_2_10, verbose=False,test_size=0.2,seed=args.random_seed)
    #decision_tree_base, svm_base=AdaBoost_crossval(dataset_training=X_2_10_train, labels_training=Y_2_10_train,kernel="rbf",C=10,gamma=0.001,max_depth=4, min_samples_leaf=2, max_leaf_nodes=5,min_samples_split=2,criterion="gini", random_state=0, data_type=f"no dim red 2_10 together, with seed {args.random_seed}",verbose=False)
    # AdaBoost classification
    #AdaBoost_classification(dataset_training=X_2_10_train, labels_training=Y_2_10_train, dataset_test=X_2_10_test,labels_test=Y_2_10_test,estimator= decision_tree_base,n_estimators=10,learning_rate=0.001, data_type=f"no dim red 2_10 together, with seed {args.random_seed}", verbose=False,see_plots=False,save_plots=True,confusion_mat=False,accuracy_per_class=True)
    # Classification with grassland and remnant forest together, with best SVMs for original dataset and DT for original dataset
    #X_G_R,Y_G_R,X_G_R_train,X_G_R_test,Y_G_R_train,Y_G_R_test = data_preprocessing_G_R(dataset_G_R=data_G_R, verbose=False,test_size=0.2,seed=args.random_seed)
    #decision_tree_base, svm_base=AdaBoost_crossval(dataset_training=X_G_R_train, labels_training=Y_G_R_train,kernel="rbf",C=10,gamma=0.001,max_depth=4, min_samples_leaf=2, max_leaf_nodes=5,min_samples_split=2,criterion="gini", random_state=0, data_type=f"no dim red G_R together, with seed {args.random_seed}",verbose=False)
    # AdaBoost classification
    #AdaBoost_classification(dataset_training=X_G_R_train, labels_training=Y_G_R_train, dataset_test=X_G_R_test,labels_test=Y_G_R_test,estimator= decision_tree_base,n_estimators=80,learning_rate=0.1, data_type=f"no dim red G_R together, with seed {args.random_seed}", verbose=False,see_plots=False,save_plots=True,confusion_mat=False,accuracy_per_class=True)
    # Classification with 2 years old plantation and remnant forest together, with best SVMs for original dataset and DT for original dataset
    X_2_R,Y_2_R,X_2_R_train,X_2_R_test,Y_2_R_train,Y_2_R_test = data_preprocessing_2_R(dataset_2_R=data_2_R, verbose=False,test_size=0.2,seed=args.random_seed)
    decision_tree_base, svm_base=AdaBoost_crossval(dataset_training=X_2_R_train, labels_training=Y_2_R_train,kernel="rbf",C=10,gamma=0.001,max_depth=4, min_samples_leaf=2, max_leaf_nodes=5,min_samples_split=2,criterion="gini", random_state=0, data_type=f"no dim red 2_R together, with seed {args.random_seed}",verbose=False)
    # AdaBoost classification
    AdaBoost_classification(dataset_training=X_2_R_train, labels_training=Y_2_R_train, dataset_test=X_2_R_test,labels_test=Y_2_R_test,estimator= decision_tree_base,n_estimators=10,learning_rate=1, data_type=f"no dim red 2_R together, with seed {args.random_seed}", verbose=False,see_plots=False,save_plots=True,confusion_mat=False,accuracy_per_class=True)