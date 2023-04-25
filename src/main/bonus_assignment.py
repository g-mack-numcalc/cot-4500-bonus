import numpy as np

def make_diagonally_dominant(matrix, b_vector):
    n = len(matrix)  # Get the size of the matrix (assumes square matrix)
    
    # Iterate through each row of the matrix
    for i in range(n):
        pivot = matrix[i][i]  # Get the pivot element (element on the diagonal)
        sum_of_other_elements = sum(abs(matrix[i][j]) for j in range(n) if j != i)  # Get the sum of absolute values of the other elements in the row
        
        # Check if the pivot is already the largest element in the row
        if abs(pivot) >= abs(sum_of_other_elements):
            continue  # If yes, move on to the next row
        
        # If not, find the row with the largest absolute value in the pivot column
        max_value_of_row = 0
        max_index_in_row = i
        for j in range(i+1, n):
            current_value_in_row = abs(matrix[j][i])
            if current_value_in_row > max_value_of_row:
                max_value_of_row = current_value_in_row
                max_index_in_row = j
                
        # Swap the current row with the row with the largest absolute value in the pivot column
        matrix[[i, max_index_in_row]] = matrix[[max_index_in_row, i]]
        b_vector[[i, max_index_in_row]] = b_vector[[max_index_in_row, i]]
        
    # Return the modified matrix and original vector as a tuple
    return matrix, b_vector

def gauss_seidel(matrix, b_vector, x_init, eps=1e-6, max_iterations=50):
    n = len(matrix)
    x_prev = x_init.copy()
    for iteration in range(max_iterations):
        x_next = np.zeros(n)
        for i in range(n):
            # Calculate the next x value for the current row using the Gauss-Seidel formula
            x_next[i] = (b_vector[i] - sum(matrix[i][j] * x_next[j] for j in range(i)) - sum(matrix[i][j] * x_prev[j] for j in range(i+1, n))) / matrix[i][i]
        # Check if the difference between the next x vector and the previous x vector is below the epsilon value
        if np.linalg.norm(x_next - x_prev) < eps:
            break
        x_prev = x_next.copy()
    print(iteration+1)
    return x_next

def jacobi(matrix, b_vector, x_init, eps=1e-6, max_iterations=50):
    n = len(matrix)
    x_prev = x_init.copy()
    for iteration in range(max_iterations):
        x_next = np.zeros(n)
        for i in range(n):
            # Calculate the next x value for the current row using the Jacobi formula
            x_next[i] = (b_vector[i] - sum(matrix[i][j]*x_prev[j] for j in range(n) if j != i)) / matrix[i][i]
        # Check if the difference between the next x vector and the previous x vector is below the epsilon value
        if np.linalg.norm(x_next - x_prev) < eps:
            break
        x_prev = x_next.copy()
    print(iteration+1)
    return x_next

def f(x):
    return x**3 - x**2 + 2

def df(x):
    return 3*x**2 - 2*x

def newton_raphson(x0, tol):
    x1 = x0 - f(x0) / df(x0)  # Calculate the first iteration
    count = 1  # Initialize the iteration count to 1

    # Continue iterating until the absolute difference between x1 and x0 is less than the tolerance
    while abs(x1 - x0) > tol:
        x0 = x1  # Update x0 with the previous x1 value
        x1 = x0 - f(x0) / df(x0)  # Calculate the next iteration
        count += 1  # Increment the iteration count

    return count  # Return the iteration count

def apply_div_dif(matrix: np.array):
    size = len(matrix)

    # Iterate through each cell in the matrix and calculate the divided differences
    for i in range(2, size):
        for j in range(2, min(i + 2, size)):
            # Skip if value is prefilled (to avoid recalculating)
            if matrix[i][j] != 0:
                continue

            # Get the left cell entry and diagonal left entry
            left: float = matrix[i][j - 1]
            diagonal_left: float = matrix[i - 1][j - 1]
            numerator: float = (left - diagonal_left)  # Calculate the numerator
            denominator = matrix[i][0] - matrix[i - j + 1][0]  # Calculate the denominator

            # If the denominator is 0, set the cell to 0 to avoid division by 0
            if denominator == 0:
                matrix[i][j] = 0
            else:
                operation = numerator / denominator  # Calculate the divided difference
                matrix[i][j] = operation  # Set the cell value to the divided difference

    return matrix  # Return the updated matrix

def hermite_interpolation():
    # Define the data points and slopes
    x_points = [0, 1, 2]
    y_points = [1, 2, 4]
    slopes = [1.06, 1.23, 1.55]

    # Initialize the matrix with zeros and fill in the x, y, and slope values
    num_of_points = len(x_points)
    matrix = np.zeros((2*num_of_points, 2*num_of_points))
    for i, x in enumerate(x_points):
        matrix[2*i][0] = x
        matrix[2*i+1][0] = x
    for i, y in enumerate(y_points):
        matrix[2*i][1] = y
        matrix[2*i+1][1] = y
    for i, slope in enumerate(slopes):
        matrix[2*i+1][2] = slope

    # Apply the Divided Differences method and print the result
    filled_matrix = apply_div_dif(matrix)
    print(filled_matrix)

def function(t: float, y: float):
    return y - t**3

def do_work(t, y, h):
    # Calculate the basic and incremented function calls and return their sum
    basic_function_call = function(t, y)
    incremented_t = t + h
    incremented_y = y + h * basic_function_call
    incremented_function_call = function(incremented_t, incremented_y)
    return basic_function_call + incremented_function_call

def modified_eulers():
    # Set the initial values for the differential equation
    original_y = 0.5
    start_of_t, end_of_t = (0, 3)
    num_of_iterations = 100

    # Calculate the step size
    h = (end_of_t - start_of_t) / num_of_iterations

    # Perform the iterative calculations using the Modified Euler's method
    for cur_iteration in range(num_of_iterations):
        # Set up the initial values for the inner function call
        t = start_of_t
        y = original_y

        # Call the do_work function to calculate the increment
        inner_math = do_work(t, y, h)

        # Calculate the next approximation using the Modified Euler's method
        next_y = y + (h / 2) * inner_math

        # Set the new initial values for the next iteration
        start_of_t = t + h
        original_y = next_y

    # Print the final approximation and return None
    print("%.5f" % original_y)
    return None

if __name__ == "__main__":
    matrix = np.array([[3, 1, 1],
                       [1, 4, 1],
                       [2, 3, 7]])
    b_vector = np.array([1, 3, 0])
    d_matrix, new_b = make_diagonally_dominant(matrix, b_vector)
    x_init = np.zeros(len(matrix))

    #1
    x_final_gauss = gauss_seidel(d_matrix, new_b, x_init)
    print("")

    #2
    x_final_jacobi = jacobi(d_matrix, new_b, x_init)
    print("")

    #3
    tol = 1e-6
    x0 = 0.5
    print(newton_raphson(x0, tol))
    print("")

    #4
    hermite_interpolation()
    print("")

    #5
    modified_eulers()