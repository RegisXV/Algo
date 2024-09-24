linear_count = 0
divide_conquer_count = 0
arr1 = [6,22,34,0,15,65,15,0,34,22,6]
arr2 = [25,6,54,16,25,12,0,25,6,54,16]
arr3 = [6,22,34,0,15,65,15,0,34,22,6,65]

def find_peak_linear(arr):
    global linear_count
    linear_count = 0
    n=len(arr)
    for i in range(n):
        linear_count += 1
        if (i==0 or arr[i]>=arr[i-1]) and (i==n-1 or arr[i]>=arr[i+1]):
            return arr[i]
    return None

def find_peak_divide_conquer(arr, low, high):
    global divide_conquer_count
    divide_conquer_count = 0
    
    def helper(arr,low,high):
        global divide_conquer_count
        mid = (low+high)//2
        divide_conquer_count += 1
        n = len(arr)
        if (mid==0 or arr[mid] >= arr[mid-1]) and (mid==n-1 or arr[mid] >= arr[mid+1]):
            return arr[mid]
        elif mid>0 and arr[mid-1]>arr[mid]:
            return helper(arr,low,mid-1)
        else:
            return helper(arr,mid+1,high)
        
    return helper(arr,low,high)


value1 = find_peak_linear(arr1)
print('linear value',value1)
print('linear count',linear_count)

value4 = find_peak_divide_conquer(arr1,0,len(arr1)-1)
print('divide conquer',value4)
print('calculations',divide_conquer_count)

value2 = find_peak_linear(arr2)
print('linear value',value2)
print('linear count',linear_count)

value5 = find_peak_divide_conquer(arr2,0,len(arr2)-1)
print('divide conquer',value5)
print('calculations',divide_conquer_count)

value3 = find_peak_linear(arr3)
print('linear value',value3)
print('linear count',linear_count)

value6 = find_peak_divide_conquer(arr3,0,len(arr3)-1)
print('divide conquer',value6)
print('calculations',divide_conquer_count)

