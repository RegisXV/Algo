array = [20,35,14,22,48,31,45,50,5,9,17,19,40,45,50,60,75,80,85,90,95,100]

def bubbleSort(array):
    for i in range(len(array)):
        print (array,"BubbleSort")
        for j in range(0, len(array)-i-1):
            if array[j] > array[j+1]:
                array[j], array[j+1] = array[j+1], array[j]
    return array


print("Sorted array is: ", bubbleSort(array))

def linearSearch(array, x):
    for i in range(len(array)):
        print(array,"LinearSearch")
        if array[i] == x:
            return i
    return -1

    
print("Element found at index: ", linearSearch(array, 50))


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

print("Element found at index: ", binarySearch(array, 50))