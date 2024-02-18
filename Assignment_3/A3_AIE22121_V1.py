#importation of the libraries
import pandas as pd # data processing and framing
import numpy as np # numpy to use few functions like  mean, min, max, matrix functions

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from datetime import datetime

import matplotlib.pyplot as plt


def question1(purchase_data):
    np_data = purchase_data.to_numpy()
    
    A = np_data[:, 1:-2] # from the first to last - 1 columns
    C = np_data[:, -2] # this is the last column
    
    dimentions = A.shape[1]  # number of features or variables in our dataset
    print (f"The dimensionality of the dataset: {dimentions}")
    
    vectors = A.shape[0] # no of rows in the dataset
    print(f"The number of vectors in the dataset: {vectors}") 
    
    A = A.astype(float) # To convert the MAtrix a into numerical values
    
    rankofA = np.linalg.matrix_rank(A) # calculate the rank of matrix
    print(f"Rank of Matrix A :{rankofA}")
    
    pseudo_inv = np.linalg.pinv(A) #Calculate the psudo inverse of the matrix A
    
    return dimentions, vectors, rankofA, pseudo_inv, C


def Question_2(pseudo_inv, C):
    product_costs = np.dot(pseudo_inv, C)  # Multiply each row of pseudo inverse with corresponding column of C
    return product_costs
    
    
def Question_3(purchase_data):
    X = purchase_data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]  # Features
    y = purchase_data['Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 42) # Split the data into training and testing sets
    
    knn = KNeighborsClassifier(n_neighbors = 6)  # Using k=3 for simplicity
    
    knn.fit(X_train, y_train) # Train the classifier
    
    y_pred = knn.predict(X_test) # Predict on the test data
    
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    return accuracy    


def Question_4():
    data = pd.read_excel('lab_3.xlsx', sheet_name = "IRCTC Stock Price") # reaidng the excel file
    
    price_mean = data["Price"].mean() # mean of the prices
    price_var = data["Price"].var()
    
    wednesday_prices = data[data["Day"] == 'Wed']["Price"] #  getting all the prices on Wednesdays
    wed_mean =  wednesday_prices.mean()   # Mean of all the prices on Wednesdays
    
    april_prices = data[data['Month'] == 'Apr']["Price"]  # Getting April's prices
    april_mean = april_prices.mean()   # Mean of all the prices in April
    
    loss_count = data[data["Chg%"] < 0]["Chg%"].count()  # Counting losses
    total_count = len(data) 
    loss_probabiity  = loss_count/total_count   # Calculating the probability of a loss stock
    
    wed_profit = data[data['Chg%'] > 0][data['Day'] == 'Wed']["Chg%"].count()  # Profitable trades made on Wednesdays
    wed_profit_probablity = wed_profit/total_count   # calculating the probability of profit if we buy on Wednesday and sell any day after
    
    total_wed = data[data['Day'] == 'Wed']['Price'].count()   # Total number of weeks with Wednesday as day
    conditional_probability = wed_profit/total_wed    # Probability of a profit given it is Wednesday
    
    # Scatter plot of Chg% data against the day of the week
    plt.scatter(data['Day'], data['Chg%'])
    plt.xlabel('Day of the Week')
    plt.ylabel('Change Percentage')
    plt.title('Change Percentage vs Day of the Week')
    plt.savefig('scatter_plot.png')
    plt.show()
        
    return wed_mean, price_mean, april_mean, loss_probabiity, wed_profit_probablity, conditional_probability
    
        
        
def main():
    print("Question 1: ")
    purchase_data = pd.read_excel('lab_3.xlsx', sheet_name = "Purchase data") # reading the excel fil e 
    purchase_data = purchase_data[['Customer', 'Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)', 'Status']] # extracting only the needed columns
    
    dim, vectors, rank_of_a, inv, C = question1(purchase_data = purchase_data)
    
    # Printing of the values
    print (f"The dimensionality of the dataset: {dim}")
    print(f"The number of vectors in the dataset: {vectors}")
    print(f"Rank of Matrix A :{rank_of_a}")
    
    
    print("Question 2: ")
    product_costs = Question_2(inv, C)
    print("Cost of each product available for sale:")
    for i, cost in enumerate(product_costs):
        print(f"Product {i+1}: {cost}")
    

    print("Question 3: ")
    accuracy = Question_3(purchase_data)
    print(f"Accuracy of the classifier: {accuracy}")
    
    print("Question 4: ")
    wed_mean, price_mean, april_mean, loss_probabiity, wed_profit_probablity, conditional_probability = Question_4()
    print(f"Mean price of Wednesdays: {wed_mean}")
    print(f"mean of the prices for all the population: {price_mean}")
    print(f"the mean of the prices for April month: {april_mean}")
    print(f"The probability that a stock will lose: {loss_probabiity}")
    print(f"The probability that a stock will make profit on Wednesday: {wed_profit_probablity}")
    print(f'The conditional probabitlity: {conditional_probability}')
    
main()