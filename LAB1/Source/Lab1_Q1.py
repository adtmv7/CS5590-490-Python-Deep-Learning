# Given a collection of integers that might contain duplicates, nums, return all possible subsets.
# Do not include null subset.

import itertools

subset_list = [] # Creating a list to store values, empty set
# Input number of elements you want to create subsets for
input_elements = int(input("Enter the number of elements you need subsets for: "))
print("Enter the number one by one you need subset for:") # Enter the integer for creating subsets
for i in range(0, input_elements): # loop through inputs, each time appending on a new value to each existing subset
    array_elements = int((input()))
    subset_list.append(array_elements) # we are adding the elements to the subset_list
print('All possible subset for given elements are:' )
for i in range(1,len(subset_list)+1):
    output_subset = (list(itertools.combinations(subset_list,i))) # Prints all possible combinations of list
    print(list(set(output_subset)))


