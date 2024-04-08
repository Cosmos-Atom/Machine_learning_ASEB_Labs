import numpy as np
import pandas as pd

# A1. Function to find the feature for the root node of a Decision Tree using Information Gain
def find_root_feature(data, labels):
    num_features = len(data.columns)
    num_samples = len(labels)
    base_entropy = calculate_entropy(labels)

    best_info_gain = 0.0
    best_feature = None

    for feature in data.columns:
        unique_values = data[feature].unique()

        new_entropy = 0.0
        for value in unique_values:
            sub_data = data[data[feature] == value]
            sub_labels = labels[data.index.isin(sub_data.index)]
            prob = len(sub_data) / float(num_samples)
            new_entropy += prob * calculate_entropy(sub_labels)

        info_gain = base_entropy - new_entropy

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = str(feature)

    return best_feature

# Function to calculate entropy
def calculate_entropy(labels):
    num_samples = len(labels)
    unique_labels = labels.unique()
    entropy = 0.0
    for label in unique_labels:
        prob = float((labels == label).sum()) / num_samples
        entropy -= prob * np.log2(prob)
    return entropy

# Function to split data based on a feature value
def split_data(data, labels, feature, value):
    sub_data = data[data[feature] == value]
    sub_labels = labels[data.index.isin(sub_data.index)]
    return sub_data, sub_labels

# A2. Function to bin continuous valued attribute to categorical valued
def binning_continuous(data, num_bins=10, binning_type='equal_width'):
    if binning_type == 'equal_width':
        return equal_width_binning(data, num_bins)
    elif binning_type == 'frequency':
        return frequency_binning(data, num_bins)
    else:
        raise ValueError("Invalid binning type. Choose 'equal_width' or 'frequency'.")

# Function for equal width binning
def equal_width_binning(data, num_bins):
    min_val = data.min()
    max_val = data.max()
    bin_width = (max_val - min_val) / num_bins
    bins = []
    for i in range(num_bins):
        bins.append((min_val + i*bin_width, min_val + (i+1)*bin_width))
    binned_data = pd.cut(data, bins, labels=False)
    return binned_data, bins

# Function for frequency binning
def frequency_binning(data, num_bins):
    binned_data, bins = pd.qcut(data, q=num_bins, retbins=True, labels=False)
    bins = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
    return binned_data, bins

# A3. Function to build Decision Tree
def build_decision_tree(data, labels, features):
    # Base cases
    if len(set(labels)) == 1: 
        return {'label': labels.iloc[0]}
    if len(features) == 0:  
        return {'label': labels.mode().iloc[0]}

    # Find the best feature to split on
    root_feature = find_root_feature(data, labels)

    # Initialize tree with the root feature
    tree = {'feature_name': root_feature, 'children': {}}

    # Remove the root feature from available features
    remaining_features = features.drop(root_feature)

    unique_values = data[root_feature].unique()
    for value in unique_values:
        sub_data, sub_labels = split_data(data, labels, root_feature, value)
        tree['children'][value] = build_decision_tree(sub_data, sub_labels, remaining_features)

    return tree

# Function to read data from CSV file using pandas
def read_csv(file_path):
    df = pd.read_csv(file_path)
    data = df.drop(columns=['rowid'])  
    labels = df.iloc[:, -1]  
    return data, labels, data.columns

file_path = 'imputed_data.csv'  
data, labels, features = read_csv(file_path)

root_feature = find_root_feature(data, labels)
print("Root feature:", root_feature)

tree = build_decision_tree(data, labels, features)
import json

def write_tree_to_file(tree, file_path):
    with open(file_path, 'w') as file:
        json.dump(tree, file)

file_path_tree = 'decision_tree.json'
write_tree_to_file(tree, file_path_tree)
