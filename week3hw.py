import random

def main():
    highest_peak_value = 0
    highest_peak_position = 0

    for i in range(10):
        # Create a list of 10 random numbers
        numbers = generate_random_numbers()
        print(numbers)
        peak_value, peak_position = find_peak_index(numbers)
        print('TruePeakPosition', peak_position, 'with value', peak_value)

def generate_random_numbers():
    # Return a list of 10 random numbers
    return [random.randint(1, 100) for _ in range(10)]

def find_peak_index(numbers):
    truevalue = 0
    truepeak = 0
    n = len(numbers)

    # Handle edge cases for the first and last elements
    if n == 0:
        return None, None
    if n == 1 or numbers[0] >= numbers[1]:
        truepeak = 1
        truevalue = numbers[0]
    if numbers[-1] >= numbers[-2]:
        truepeak = n
        truevalue = numbers[-1]

    # Check for a peak in the rest of the array
    for i in range(1, n - 1):
        if numbers[i] >= numbers[i - 1] and numbers[i] >= numbers[i + 1] and numbers[i] >= truevalue:
            truevalue = numbers[i]
            truepeak = i + 1  # Convert to positional index

    return truevalue, truepeak

main()