import cv2
import numpy as np


def standardize_input(image):
    #Resize all images to be 32x32x3
    standard_im = cv2.resize(image,(32,32))
    return standard_im


'''
HSV即色相、饱和度、明度（英语：Hue, Saturation, Value），又称HSB，其中B即英语：Brightness。
色相（H）是色彩的基本属性，就是平常所说的颜色名称，如红色、黄色等。
饱和度（S）是指色彩的纯度，越高色彩越纯，低则逐渐变灰，取0-100%的数值。
明度（V），亮度（L），取0-100%。
此py文件的主要目的是将trafficlights的颜色进行变换，并且给出结论以备使用
'''


def estimate_label(rgb_image, display=False):
    '''
    rgb_image:Standardized RGB image
    '''
    colors = ['Red_traffic_Light', 'Green_traffic_Light', 'Yellow_traffic_Light']
    rgb_image = standardize_input(rgb_image)
    a = red_green_yellow(rgb_image, display).index(1)
    return colors[a]

def findNoneZero(rgb_image):
    rows, cols, _ = rgb_image.shape
    counter = 0
    for row in range(rows):
        for col in range(cols):
            pixels = rgb_image[row, col]
            if sum(pixels) != 0:
                counter = counter + 1
    return counter


def red_green_yellow(rgb_image, display):
    '''
    Determines the red , green and yellow content in each image using HSV and experimentally
    determined thresholds. Returns a Classification based on the values
    '''
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    sum_saturation = np.sum(hsv[:, :, 1])  # Sum the brightness values
    area = 32 * 32
    avg_saturation = sum_saturation / area  # find average

    sat_low = int(avg_saturation * 1.3)  # 均值的1.3倍，工程经验
    val_low = 140
    # Green
    lower_green = np.array([70, sat_low, val_low])
    upper_green = np.array([100, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_result = cv2.bitwise_and(rgb_image, rgb_image, mask=green_mask)
    # Yellow
    lower_yellow = np.array([10, sat_low, val_low])
    upper_yellow = np.array([60, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_result = cv2.bitwise_and(rgb_image, rgb_image, mask=yellow_mask)

    # Red
    lower_red = np.array([150, sat_low, val_low])
    upper_red = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    red_result = cv2.bitwise_and(rgb_image, rgb_image, mask=red_mask)
    sum_green = findNoneZero(green_result)
    sum_red = findNoneZero(red_result)
    sum_yellow = findNoneZero(yellow_result)
    if sum_red >= sum_yellow and sum_red >= sum_green:
        return [1, 0, 0]  # Red
    if sum_yellow >= sum_green:
        return [0, 1, 0]  # yellow
    return [0, 0, 1]  # green
