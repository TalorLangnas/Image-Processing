"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List

import numpy as np
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def myID() -> int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 204240477


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    if representation == 1:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = np.asarray(img, dtype=np.float32)
        img = img * (1 / 255.0)
        return img

    elif representation == 2:
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, dtype=np.float32)
        img = img * (1 / 255.0)
        return img

def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    if representation == 1:
        img_arr = imReadAndConvert(filename, representation)
        plt.imshow(img_arr, cmap='gray')
        plt.show()
    elif representation == 2:
        img_arr = imReadAndConvert(filename, representation)
        plt.imshow(img_arr)
        plt.show()
    else:
        raise Exception("representation have to be 1 or 2")

    return None


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    transformer = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    img_YIQ = np.dot(imgRGB, transformer.transpose())

    return img_YIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    transformer = np.array([[1.000, 0.956, 0.619], [1.000, -0.272, -0.647], [1.000, -1.106, 1.703]])
    imgRGB = np.dot(imgYIQ, transformer.transpose())

    return imgRGB

def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    if len(imgOrig.shape) == 3:
        imgOrig_YIQ = transformRGB2YIQ(imgOrig)
        imEq_YIQ, histOrg, histEq = hsitogramEqualize_2D(imgOrig_YIQ[:, :, 0])
        imgOrig_YIQ[:, :, 0] = imEq_YIQ
        imEq = transformYIQ2RGB(imgOrig_YIQ)

    else:
        imEq, histOrg, histEq = hsitogramEqualize_2D(imgOrig)

    return imEq, histOrg, histEq


def hsitogramEqualize_2D(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig_256: Original Histogram
        :ret
    """

    imgOrig_256 = cv2.normalize(imgOrig, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


    # 1. calculating histogram (value of [0,255])
    histOrg, bins_ = np.histogram(imgOrig_256.flatten(), 256, [0, 256])
    # calculating total number of pixels
    total_pixels = np.sum(histOrg)

    # 2. calculating PDF (value of [0,255])
    histOrg_PDF = histOrg * (1/total_pixels)

    # 3. calculating CDF (value of [0,255])
    normalized_CDF = np.cumsum(histOrg_PDF)

    # 4. creating LUT table
    new_color = np.zeros_like(normalized_CDF)
    for i in range(len(new_color)):
        new_color[i] = 255*(normalized_CDF[i])
    new_color = np.floor(new_color)

    # 5. mapping the intensity values to the image
    imEq = np.zeros_like(imgOrig_256)
    for i in range(len(new_color)):
        imEq[imgOrig_256 == i] = new_color[i]

    # 6. calculating the histogram of imEq (value of [0,255])
    histEq, bins_ = np.histogram(imEq.flatten(), 256, [0,256])
    # 7. normalized imEq's pixels to [0,1]
    imEq = imEq * (1/255.0)

    return imEq, histOrg, histEq


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    qImage_i = []
    error_i = []

    # if imOrig is RGB we convert to YIQ and perform same procedure as GRAY_IMAGE
    if len(imOrig.shape) == 3:
        imOrig_YIQ = transformRGB2YIQ(imOrig)
        img_2D_org = imOrig_YIQ[:, :, 0]

    else:
        # image is GRAY
        img_2D_org = np.copy(imOrig)
    # normalize the values to [0,255]:
    img_2D = cv2.normalize(img_2D_org, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    hist, bins_ = np.histogram(img_2D.flatten(), 256, [0, 256])
    z_list = np.linspace(img_2D.min(), img_2D.max(), nQuant + 1, endpoint=True, dtype=np.int32)  # [1:-1]

    for i in range (nIter-1):
        # find the values for each interval
        q_list = find_q(hist, z_list)
        # find the bounds for nQuant
        z_list = find_z(q_list)
        # mapping the new values to the image
        img_tmp = mapping_pix(img_2D, q_list, z_list)
        img_tmp = img_tmp/255

        error_i.append((mean_squared_error(img_2D_org, img_tmp)))
        qImage_i.append(img_tmp)

        # stop the process if mean error converges
        if len(error_i) > 1:
            if abs(error_i[i] - error_i[i-1]) < 0.0000001:
                break

    if len(imOrig.shape) == 3:
        qImage_tmp = []
        for img in qImage_i:
            imOrig_YIQ[:, :, 0] = img
            img_tmp = transformYIQ2RGB(imOrig_YIQ)
            qImage_tmp.append(img_tmp)

        qImage_i = qImage_tmp

    return qImage_i, error_i


def mapping_pix(img: np.ndarray ,values_list: np.ndarray, bounders_list: np.ndarray) -> (np.ndarray):

    """
        Mapping values to np.ndarray
        :param img: The original image (RGB or Gray scale)
        :param values_list: pixels values
        :param bounders_list: bounders list
        :return: (np.ndarray: updated image)
    """

    new_img = np.zeros_like(img)
    start = 0
    end = start + 1

    for i in range(len(values_list)):
        pix_val = (img >= bounders_list[start]) & (img < bounders_list[end]+ 1)
        new_img[pix_val] = values_list[start]
        start += 1
        end += 1

    return new_img

def find_q(histogram: np.ndarray, z_list: np.ndarray) -> (np.ndarray):

    start = 0
    end = start + 1
    q_list = np.array([[]])

    for i in range(1,len(z_list)):
         values = np.arange(z_list[start], z_list[end] + 1)
         if np.sum(histogram[z_list[start]:z_list[end] + 1]) == 0:
             q_list = np.append(0)
         else:
            sub_histogram_PDF = np.divide((histogram[z_list[start]:z_list[end] + 1]), np.sum(histogram[z_list[start]:z_list[end] + 1]))
            q_list = np.append(q_list, np.dot(values, sub_histogram_PDF))

         start += 1
         end += 1

    return q_list

def find_z(q_list: np.ndarray) -> (np.ndarray):
    z_list = np.array([[]])
    start = 0
    end = start + 1

    # finding the new bounds:
    for i in range(1,len(q_list)):
        z_list = np.append(z_list, (q_list[start] + q_list[end]) / 2)
        start += 1
        end += 1

    # insert 0 and 255 to z_list:
    z_list = np.insert(z_list, 0, 0)
    z_list = np.append(z_list, 255)
    # convert z_list to int32 nparray
    return z_list.astype(int)



