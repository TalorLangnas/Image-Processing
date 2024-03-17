import math
import numpy as np
import cv2



def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """

    return 204240477

def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """

    if len(in_signal) < len(k_size):
        signal = k_size
        kernel = in_signal
    else:
        signal = in_signal
        kernel = k_size
    padd = len(kernel) - 1
    reversed_kernel = kernel[::-1]
    padding_arr = np.pad(signal, (padd, padd), mode='constant')
    output_arr = np.zeros(len(signal) + len(kernel) - 1)
    for i in range(len(output_arr)):
        output_arr[i] = np.sum(padding_arr[i:i + len(kernel)] * reversed_kernel)

    return output_arr


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """

    # input validation:
    if (in_image.size < kernel.size):
        image = kernel
        ker = in_image
    else:
        image = in_image
        ker = kernel

    # padding the input image:
    image_height, image_width = image.shape
    ker_height, ker_width = ker.shape
    padding = [ker_height // 2, ker_width // 2]
    padding_arr = np.pad(image, ((padding[0], padding[0]), (padding[1], padding[1])), 'edge')

    # flipping the kernel:
    flipped_ker = np.flip(ker)
    output_arr = np.zeros_like(image)
    # Computing the convolution:
    for i in range(image_height):
        for j in range(image_width):
            output_arr[i, j] = np.sum(padding_arr[i: i + ker_height, j:j + ker_width] * flipped_ker)#.round()

    return output_arr


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """

    _kernel = np.array([[-1, 0, 1]])
    x_der = conv2D(in_image, _kernel)
    y_der = conv2D(in_image, _kernel.T)

    magnitude = np.sqrt(np.square(x_der) + np.square(y_der)).astype(np.float64)
    orientation = np.arctan2(y_der, x_der).astype(np.float64)

    return orientation, magnitude


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    if k_size % 2 != 0:
        k = k_size
    else:
        k = k_size

    kernel = get_gaussian_kernel(k)
    img = cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.float64)

    return img


def pascal_triangle_row(n):
    row = [1]  # Initialize the first row with a single 1

    for _ in range(n):
        # Generate the next row based on the current row
        row = [1] + [row[i] + row[i + 1] for i in range(len(row) - 1)] + [1]

    return row

def get_gaussian_kernel(k: int) -> np.ndarray:
    coefficients = pascal_triangle_row(k - 1)
    coefficients = np.array(coefficients)
    sum_coeff = coefficients.sum()
    ker = np.outer(coefficients, coefficients.T)
    val = 1 / sum_coeff
    print(f"kernel =\n {val} * {ker}")
    gaussian_kernel = 1 / sum_coeff * (np.outer(coefficients, coefficients.T))

    return gaussian_kernel


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    if k_size % 2 != 0:
        k = k_size
    else:
        k = k_size
    sigma = math.sqrt(((k - 1) / 2) / 2)
    kernel = cv2.getGaussianKernel(k, sigma)
    img = cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.float64)

    kernel_sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8

    print("sigma = ", sigma)
    print("kernel_sigma = ", kernel_sigma)
    return img

def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    # 2nd order derivative kernel:
    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]
                          ])

    img_tmp = cv2.filter2D(img, -1, laplacian, borderType=cv2.BORDER_REPLICATE)#.astype(np.float64)
    zero_crossing_mat = get_zero_crossing(img_tmp)
    return zero_crossing_mat

def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    return edgeDetectionZeroCrossingSimple(img_blur)

def get_zero_crossing(img: np.ndarray) -> np.ndarray:
    trash_hold = -20
    img = img * 255
    row, col = np.shape(img)
    zero_cross = np.zeros_like(img)

    # checking the crossing point # checking the crossing point
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            a = img[i - 1, j] * img[i + 1, j]
            b = img[i, j - 1] * img[i, j + 1]

            # case 1: pixel is equal to zero
            if img[i, j] == 0:
                zero_cross[i, j] = 255

            # as the trash hold is lower the sensitivity gor edges is higher
            elif a < trash_hold or b < trash_hold:
                # if true its mean that we're crossing the zero
                zero_cross[i, j] = 255

    zero_cross = (zero_cross * 1/255)
    return zero_cross

def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """
    thresh_hold = 90
    circles = []
    skips = 1
    min_radius += 6
    number = 5
    edge_img = cv2.Canny((img * 255).astype(np.uint8), img.shape[0], img.shape[1])
    if max_radius - min_radius > 30:
        skips = 2
        min_radius -= 6
        number = 4

    for radius in range(min_radius, max_radius, skips):
        # print("current radius to check: {}".format(radius))
        possible_center = search_point(img, edge_img, thresh_hold, radius, number)
        if np.amax(possible_center) > thresh_hold:
            for i in range(1, int(img.shape[0] / 2) - 1):
                _i = i * 2
                for j in range(1, int(img.shape[1] / 2) - 1):
                    _j = j * 2
                    if possible_center[_i][_j] >= thresh_hold:
                        flag = check_range(circles, i, j)
                        sum_neighbors = possible_center[-1 + _i:2 + _i, _j - 1: _j + 2].sum() / 9
                        if flag and sum_neighbors >= thresh_hold / 9:
                            possible_center[_i - radius:_i + radius, _j - radius: _j + radius] = 0
                            circles.append((_j, _i, radius))

    return circles


def search_point(img: np.ndarray, edge_img: np.ndarray, thresh, radius, val) -> np.ndarray:
    acc_mat = np.zeros(edge_img.shape)
    for x in range(0, int(img.shape[0] / 2)):
        for y in range(int(img.shape[1] / 2)):
            if edge_img[x * 2][y * 2] == 255:
                for alpha in range(180):
                    alpha_1 = int(y * 2 - math.cos(alpha * np.pi / thresh) * radius)
                    alpha_2 = int(x * 2 - math.sin(alpha * np.pi / thresh) * radius)
                    if 0 <= alpha_1 < img.shape[1] and 0 <= alpha_2 < img.shape[0]:
                        acc_mat[alpha_2][alpha_1] += val

    return acc_mat


def check_range(fcircles: list, x_1, y_2) -> bool:

    for x, y, z in fcircles:
        if np.square(y_2 * 2 - x) + np.square(x_1 * 2 - y) <= np.square(z):
            return False

    return True

def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    opencv_img = cv2.bilateralFilter(in_image, k_size,sigma_space, sigma_color )

    w_img = np.zeros_like(in_image)


    in_image = cv2.copyMakeBorder(in_image, k_size, k_size, k_size, k_size,
                                  cv2.BORDER_REPLICATE, None, value=0)
    for y in range(k_size, in_image.shape[0] - k_size):
        for x in range(k_size, in_image.shape[1] - k_size):
            # same as we saw in the tirgul:
            pivot_v = in_image[y, x]
            neighbor_hood = in_image[
                            y - k_size:y + k_size + 1,
                            x - k_size:x + k_size + 1
                            ]
            sigma = sigma_space
            diff = neighbor_hood.astype(int) - pivot_v
            diff_gau = np.exp(-np.power(diff, 2) / (2 * sigma))
            gaus = cv2.getGaussianKernel(2 * k_size + 1, sigma=sigma_color)
            gaus = gaus.dot(gaus.T)
            combo = gaus * diff_gau

            # sum and normalize
            result = ((combo * neighbor_hood / combo.sum()).sum())

            w_img[y - k_size, x - k_size] = round(result)

    return opencv_img, w_img







