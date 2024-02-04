def question_1(li, target_sum): # takes the arguments of list and targeted sum
    no_of_pairs = 0 
    list_for_visiting = [] #The idea  behind this is to have a list to keep adding the visited numbers 
    # This method is the best way to approach giving O(n) time complexity. Which I've learned on Geeks for geeks
    for i in li:
        if (target_sum - i) in list_for_visiting:  # here i took target sum - i gives the number we need to search. in the sense the second number of the pair
            no_of_pairs += 1
    
        list_for_visiting.append(i) # add the number to list, at the end of the loop, every number will be added and checked if it has a pair
    
    return no_of_pairs

# This fucntion is to take input of a list from user with asking no of elements
def input_list():
    n = int(input("Enter the number of elements you want in your list : "))
    lis = []
    print("Enter ",n," real numbers separated by space")
    elements = list(map(int,input().split()))
    return elements


def question_2(li):
    length_of_list = len(li)
    minimum_number = float("inf") #  initialized as infinity so that any number can become smaller than it.
    maximum_number = float("-inf") #  inirialized as -inf so that any real number can be greater than it.
    
    # For loop to take check all numbers one by one. Time complexity is O(n)
    for i in li:
        if i < minimum_number:
            minimum_number = i
        if i > maximum_number:
            maximum_number = i
    
    # Rage is the difference between the min and max of the list
    range = maximum_number - minimum_number
    return range
    

def question_3(matrix, exponent): 
    size = len(matrix)
    # Check if the matrix is square
    if len(matrix[0]) != size:
        return "Non-square matrix"
    
    # I've created another function inside to make it easier and disctict to multiply matriceswhy
    def matrix_multiply(a, b):
        return [[sum(x * y for x, y in zip(row_a, col_b)) for col_b in zip(*b)] for row_a in a]

    # The identity matrix to start multiplying the matrix itself
    result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]

    # This method gives O(n^3 * m) 3  loops for matrix multiplication and m times the power multiplication
    # For loop to get all the powers till given power
    for _ in range(exponent):
        result = matrix_multiply(result, matrix)

    return result

from collections import defaultdict

def question_4(str):
    count_dict = defaultdict(int) # A dictionary to store  each character and its frequency
    
    for char in str:
        if char.isalpha(): # if the character is alphabet then increment the value fo the key(letter)
            count_dict[char] += 1
            
    max_char = max(count_dict, key = count_dict.get) # grtting the character with the highest occurencee
    
    max_count = count_dict[max_char] # extracting the value of the alphabet

    return max_char, max_count

def main():
    print("Question 1: ")
    question1_input = [2, 7, 4, 1, 3, 6,]
    sol_1 = question_1(question1_input, 10)
    print(f"The list given is {question1_input}. \nNumber of pairs with a sum equal to the target value of 10 are {sol_1}")
    
    print("Question 2: ")
    question2_input = input_list()
    sol_2 = question_2(question2_input)
    print(f"The range of the list {question2_input} is {sol_2}")
    
    print("Question 3: ")
    question_3_input_matrix = [[1, 2], [3, 4]]
    exponent_q3 = 3
    result_matrix = question_3(question_3_input_matrix, exponent_q3)
    for row in result_matrix:
        print(row)
        
    print("Question 4: ")
    input_string = "hippopotamus"
    max_char, max_count = question_4(input_string)
    print(f"The maximally occurring character is '{max_char}' with occurrence count {max_count}.")

main()
