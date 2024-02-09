# importations of libraries
import math # for maths functions like sqrt
import pandas as pd # to work of dataprocessing, data split
import numpy as np # numpy to get accurate euclidean_distance 

def euclidean_distance(vector1, vector2): # question 1. finding the euclidean distance, lan be called as l2 norm
    length_1 = len(vector1)
    length_2 = len(vector2)
    
    if length_1 != length_2: # checking if the dimensions of vectors are same or not, if not we cant find the euclidean distance and raice an exception
        raise ValueError("Dimensions are not matching!!")
    
    sum_of_squares = 0 # formula for 2 dim --> sqrt((x1 ^ 2 - x1^2) + (y2^2- y1^2))
    for i in range(length_1):
        sum_of_squares += (vector1[i] - vector2[i]) ** 2  # sum of squares to store
    
    euclidean_dist = math.sqrt(sum_of_squares) # squre root for the sum of squares
    return euclidean_dist


def manhattan_distance(vector1, vector2): # question 2. finding the manhattan distance, lan be called as l2 norm
    length_1 = len(vector1)
    length_2 = len(vector2)
    
    if length_1 != length_2: # checking if the dimensions of vectors are same or not, if not we cant find the euclidean distance and raice an exception
        raise ValueError("Dimensions are not matching!!")
    
    sum_of_differences = 0 # formula for 2 dim --> sqrt(|(x1 - x1)| + |(y2- y1)|)
    for i in range(length_1):
        sum_of_differences += abs(vector1[i] - vector2[i])
        
    return sum_of_differences


def euclidean_distance_numpy(instance1, instance2):
    l2_norm = np.linalg.norm(instance1 - instance2)
    return l2_norm

def knn_classify(training_data, test_instance, k): # Question - 3: KNN classification
    distances = []  # list to save distance between each point
    
    for index, row in training_data.iterrows():
        dist = euclidean_distance_numpy(row[:-1], test_instance) # Euclidean distance between the current training data point and the test instance
        distances.append((row, dist))
        
    distances.sort(key=lambda x: x[1]) # Sort the distances list on the distance in ascending order
    neighbors = distances[:k]
    class_votes = {}
    
    # counting the votes for each class label 
    for neighbor in neighbors:
        label = neighbor[0][-1]  # Extracting the class label of the neighbor
        if label in class_votes:
            class_votes[label] += 1
        else: # if label is seen for the first time
            class_votes[label] = 1
            
    return max(class_votes, key = class_votes.get) # class label with the maximum number of votes
    

def label_encoding(data):
    encoded_data = data.copy() # a copy of the original data to avoid changing it directly
    label_encoders = {}
    
    for column in encoded_data.columns:
        if encoded_data[column].dtype == 'object': # if the column contains object (string) data type
            labels = encoded_data[column].unique()  
            label_encoder = {label: i for i, label in enumerate(labels)} # Create a label encoder mapping each label to a unique integer
            encoded_data[column] = encoded_data[column].map(label_encoder) # Map labels in the column to the integer values using the label encoder
            label_encoders[column] = label_encoder
    
    return encoded_data, label_encoders


def one_hot_encoding(data):
    encoded_data = data.copy() # a copy of the original data to avoid changing it directly
    one_hot_encoded_columns = [] 
    
    for column in encoded_data.columns:
        if encoded_data[column].dtype == 'object':  # if the column contains object (string) data type
            unique_values = encoded_data[column].unique()
            for value in unique_values:
                new_column_name = f"{column}_{value}"   # Create a new column name by appending the original column name and the value
                encoded_data[new_column_name] = (encoded_data[column] == value).astype(int)  # Create a new binary column indicating
                one_hot_encoded_columns.append(new_column_name)
            encoded_data.drop(column, axis=1, inplace=True)
    
    return encoded_data, one_hot_encoded_columns



def main():
    vector1 = [3, 4, 5]
    vector2 = [2, 3, 4]
    
    print("Question 1: ")
    euclidean = euclidean_distance(vector1, vector2)
    print(f"Euclidean Distance between vector 1 and vector 2 is: {euclidean}")
    manhattan = manhattan_distance(vector1, vector2)
    print(f"Manhattan Distance between vector 1 and vector 2 is {manhattan}")
    
    print("Question 2: ")
    data = pd.read_csv('apple_quality.csv')
    train_data = data.iloc[:3000]  
    test_data = data.iloc[3000:]   
    test_instance = test_data.iloc[0, :-1]  # first instance from test data
    k = 5  
    predicted_class = knn_classify(train_data, test_instance, k)
    print("Predicted class:", predicted_class)
    
    print("Question 3: ")
    categorical_data = data.select_dtypes(include=['object'])
    encoded_data, label_encoders = label_encoding(categorical_data) # label encoding
    print(label_encoders) # label encoders
    
    print("Question 4: ")
    categorical_data = data.select_dtypes(include=['object'])
    encoded_data, one_hot_encoded_columns = one_hot_encoding(categorical_data) # one-hot encoding
    print(one_hot_encoded_columns)     # one-hot encoded columns

        
main()
    