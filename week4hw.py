arr1 = [20,35,14,22,48,31,45,50,5,9,17,19,40,45,50,60,75,80,85,90,95,100]
arr2 = [100,95,90,85,80,75,60,50,45,40,19,17,9,5,50,45,31,48,22,14,35,20]
arr3 = [10,25,40,35,22,21,55,100,99,65,74,26,80,90,95,100,85,75,60,50,45,31,48,22,14,35,20]

def bubbleSort(array):
    for i in range(len(array)):
        for j in range(0, len(array)-i-1):
            if array[j] > array[j+1]:
                array[j], array[j+1] = array[j+1], array[j]
        print("step",array)
    print("After Pass",i+1,":",array)
    print('\n')
    return array


def selectionSort(array):
    n = len(array)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if array[j] < array[min_idx]:
                min_idx = j
        
        array[i], array[min_idx] = array[min_idx], array[i]
        print("After iteration",i+1,":",array)

    return array



print("Original array is: ", arr1)
print("\n")
bubble1 = bubbleSort(arr1[:])  # Using slicing to create a copy
print("Bubble Sorted array is: ", bubble1)
print("\n")
selection1 = selectionSort(arr1[:])  # Using slicing to create a copy
print("Selection Sorted array is: ", selection1)
print("\n")
print("Original array is: ", arr1)
print("\n")
print("Original array is: ", arr2)
print("\n")
bubble2 = bubbleSort(arr2[:])  # Using slicing to create a copy
print("Bubble Sorted array is: ", bubble2)
print("\n")
selection2 = selectionSort(arr2[:])  # Using slicing to create a copy
print("Selection Sorted array is: ", selection2)
print("\n")
print("Original array is: ", arr3)
print("\n")
bubble3 = bubbleSort(arr3[:])  # Using slicing to create a copy
print("Bubble Sorted array is: ", bubble3)
print("\n")
selection3 = selectionSort(arr3[:])  # Using slicing to create a copy
print("Selection Sorted array is: ", selection3)
print("\n")

