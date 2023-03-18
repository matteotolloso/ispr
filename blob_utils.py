import numpy as np
import blob_utils
from numba import njit
import matplotlib.pyplot as plt
import numpy as np
import cv2
import blob_utils


def get_kernel(fun, sigma : float, size : int) -> np.array:
    
    ker = np.empty((size, size), dtype=float)
    center_cell = size//2
    
    for ((x, y) , _) in np.ndenumerate(ker):
        ker[x , y] = fun(x - center_cell , y - center_cell , sigma)
    
    return ker

@njit
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

@njit
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

@njit
def local_min_max(matrix : np.array, center : tuple[int, int], radius : int):
    
    left_radius = max(0, center[0] - radius)
    right_radius = min(matrix.shape[0], center[0] + radius)
    up_radius = max(0, center[1] - radius)
    down_radius = min(matrix.shape[1], center[1] + radius)

    maximum = np.max(matrix[left_radius:right_radius, up_radius:down_radius])
    minumum = np.min(matrix[left_radius:right_radius, up_radius:down_radius])

    return minumum, maximum

@njit
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

@njit
def get_centers(convolved_image, blob_radius, kernel_size, percentile ):
    
    centers = []

    lower_treshold = percentile[0]
    upper_treshold = percentile[1]

    for i in range(convolved_image.shape[0]):
        for j in range(convolved_image.shape[1]):
            
            local_minimum, local_maximum = blob_utils.strictly_local_min_max(convolved_image, center=(i, j), radius=blob_radius)
            
            if (convolved_image[i,j] < local_minimum) and (convolved_image[i, j] < lower_treshold):
                centers.append( ( i + kernel_size//2, j + kernel_size//2, convolved_image[i][j], "min"))
            if (convolved_image[i,j] > local_maximum) and (convolved_image[i, j] > upper_treshold):
                centers.append( ( i+ kernel_size//2, j + kernel_size//2, convolved_image[i][j], "max"))

    return centers


def full_pipeline(path, kernel_size, sigma, percentile):
    
    def LoG(x, y, sigma) -> float:
        pi = np.pi
        return ((- 1 / (pi * sigma**4) ) * (1 - (x**2 + y**2) / (2 * sigma**2) ) ) * np.exp(-( (x**2 + y**2) / (2 * sigma**2)))
    
    kernel = blob_utils.get_kernel(fun=LoG, size=kernel_size, sigma=sigma)
    rgb_image = cv2.imread(path)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY) 
    gray_image = np.interp(
        gray_image, 
        (gray_image.min(), gray_image.max()), 
        (-1, 1)
    )
    convolved_image = blob_utils.convolve_image(
        padded_image=gray_image, 
        kernel=kernel
    )
    blob_radius = int (np.sqrt(2) * sigma) + 1
    centers = blob_utils.get_centers(
        convolved_image, 
        blob_radius=blob_radius, 
        kernel_size=kernel.shape[0], 
        percentile=np.percentile(convolved_image, [percentile[0], percentile[1]])
    )

    for (i, j , _, tipo) in centers:
        if tipo=="min":
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)   
        cv2.circle(
            rgb_image, 
            (j, i), 
            blob_radius, 
            color, 
            thickness=2, 
            lineType=2
        )

    plt.imshow(rgb_image, cmap="gray")