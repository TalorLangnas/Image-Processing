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
from ex1_utils import LOAD_GRAY_SCALE
import cv2
import numpy as np

def adjust_gamma(image: np.ndarray, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inverse_Gamma = 1.0 / gamma
    table = np.array([pow((i / 255.0), inverse_Gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    if rep == 1:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_path)

    def set_gamma(val):
        gamma = val / 100
        gamma = max(0.01, gamma)
        gamma = min(2.0, gamma)

        look_up = adjust_gamma(img, gamma)
        cv2.imshow('Gamma bar', look_up)

    cv2.namedWindow('Gamma bar')
    cv2.createTrackbar('Gamma', 'Gamma bar', 100, 200, set_gamma)

    cv2.imshow('Gamma bar', img)
    cv2.waitKey(0)


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
