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



def linearSearch(array, x):
    for i in range(len(array)):
        print(array,"LinearSearch")
        if array[i] == x:
            return i
    return -1

    



def binarySearch(array, x):
    low, high = 0, len(array) - 1
    mid = 0
    while low <= high:
        print(low,mid,high,"BinarySearch")
        mid = (low + high) // 2
        if array[mid] == x:
            return mid
        elif array[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1




print("Bubble Sorted array is: ", bubbleSort(arr1))
print("\n")
print("Linear Element found at index: ", linearSearch(arr1, 50))
print("\n")
print("Binary Element found at index: ", binarySearch(arr1, 50))
print("\n")
print("Bubble Sorted array is: ", bubbleSort(arr2))
print("\n")
print("Linear Element found at index: ", linearSearch(arr2, 50))
print("\n")
print("Binary Element found at index: ", binarySearch(arr2, 50))
print("\n")
print("Bubble Sorted array is: ", bubbleSort(arr3))
print("\n")
print("Linear Element found at index: ", linearSearch(arr3, 50))
print("\n")
print ("Binary Element found at index: ", binarySearch(arr3, 50))