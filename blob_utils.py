import numpy as np


def get_kernel(fun, sigma : float, size : int) -> np.array:
    
    ker = np.empty((size, size), dtype=float)
    center_cell = size//2
    
    for ((x, y) , _) in np.ndenumerate(ker):
        ker[x , y] = fun(x - center_cell , y - center_cell , sigma)
    
    return ker


def pad_image(image : np.array, kernel_size : int):
    
    original_rows, original_cols = image.shape

    kernel_radius = kernel_size//2

    top_left = np.zeros(shape=(kernel_radius, kernel_radius))
    top_center = np.zeros(shape=(kernel_radius, original_cols))
    top_right = np.zeros(shape=(kernel_radius, kernel_radius))
    left = np.zeros(shape=(original_rows, kernel_radius))
    right= np.zeros(shape=(original_rows, kernel_radius))
    bottom_left = np.zeros(shape=(kernel_radius, kernel_radius))
    bottom_center=np.zeros(shape=(kernel_radius, original_cols))
    bottom_right=np.zeros(shape=(kernel_radius, kernel_radius))

    padded_image =np.block([[top_left,      top_center,     top_right], 
                            [left,          image,          right],
                            [bottom_left,   bottom_center,  bottom_right]]
    )

    return padded_image

def convolve_image(padded_image: np.array, kernel: np.array):
    
    kernel_size = kernel.shape[0]
    kernel_radius = kernel_size // 2
    
    original_row= padded_image.shape[0] - kernel_radius*2
    original_col = padded_image.shape[1] - kernel_radius*2
    
    convolved_image = np.zeros(shape=(original_row, original_col))

    for i in range(original_row):
        for j in range(original_col): 
            convolved_image[i , j] = np.sum(np.multiply(
                padded_image[i : i + kernel_size , j : j + kernel_size] ,
                kernel
            ))
    
    return convolved_image


def local_min_max(matrix : np.array, center : tuple[int, int], radius : int):
    
    left_radius = max(0, center[0] - radius)
    right_radius = min(matrix.shape[0], center[0] + radius)
    up_radius = max(0, center[1] - radius)
    down_radius = min(matrix.shape[1], center[1] + radius)

    maximum = np.max(matrix[left_radius:right_radius, up_radius:down_radius])
    minumum = np.min(matrix[left_radius:right_radius, up_radius:down_radius])

    return minumum, maximum

def strictly_local_min_max(matrix : np.array, center : tuple[int, int], radius : int):
    
    left_radius = max(0, center[0] - radius)
    right_radius = min(matrix.shape[0], center[0] + radius + 1)
    up_radius = max(0, center[1] - radius)
    down_radius = min(matrix.shape[1], center[1] + radius + 1)

    center_value = matrix[center[0], center[1]]

    matrix[center[0], center[1]] = -1e6
    maximum = np.max(matrix[left_radius:right_radius, up_radius:down_radius])
    
    matrix[center[0], center[1]] = 1e6
    minumum = np.min(matrix[left_radius:right_radius, up_radius:down_radius])

    matrix[center[0], center[1]] = center_value

    return minumum, maximum