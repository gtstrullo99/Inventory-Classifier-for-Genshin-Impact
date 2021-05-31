#Alumno 1: Xiang Lin
#Alumno 2: Gerard Trullo

#Niu 1: 1465728
#Niu 2: 1494246

import cv2
import numpy as np
import matplotlib.pyplot as plt


def cropImage(path):
    img = cv2.imread(path)
    #cv2.imshow('test',img)
    #cv2.waitKey(0)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #img = (255 - img)
    img_denoised = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
    #img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    brightness = 0
    contrast = 127
    img_denoised = np.int16(img_denoised)
    img_denoised = img_denoised * (contrast / 127 + 1) - contrast + brightness
    img_denoised = np.clip(img_denoised, 0, 255)
    img_denoised = np.uint8(img_denoised)



    img_gray = cv2.cvtColor(img_denoised, cv2.COLOR_RGB2GRAY)


    #laplacian = cv2.Laplacian(img_gray, cv2.CV_64F, 3)
    #laplacian = cv2.convertScaleAbs(laplacian)
    canny = cv2.Canny(img_denoised, 200, 400)

    thresh = cv2.adaptiveThreshold(canny, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 5)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    kernel_w = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morphed = cv2.erode(thresh, kernel_h, iterations=3)
    morphed = cv2.erode(morphed, kernel_w, iterations=4)


    contours, hierarchy = cv2.findContours(morphed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    morphed_rgb = cv2.cvtColor(morphed, cv2.COLOR_GRAY2RGB)

    final = morphed_rgb.copy()

    counter = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 8000 and area < 70000:
            areas.append(cv2.contourArea(cnt))
            x, y, w, h = cv2.boundingRect(cnt)
            final = cv2.rectangle(final, (x, y), (x + w, y + h), (0, 255, 0), 2)

            item = img[y:y + h, x:x + w]
            cv2.imwrite('./train/item_{}.jpg'.format(counter),  cv2.cvtColor(item, cv2.COLOR_RGB2BGR))
            counter += 1

    # plt.axis('off')
    # plt.imshow(img)
    # plt.title('source')
    # plt.figure()
    # plt.axis('off')
    # plt.imshow(img_denoised)
    # plt.title('denoised')
    # plt.figure()
    # plt.axis('off')
    # plt.imshow(canny)
    # plt.title('canny')
    # plt.figure()
    # plt.axis('off')
    # plt.imshow(thresh)
    # plt.title('threshold')
    # plt.figure()
    # plt.axis('off')
    # plt.imshow(morphed_rgb)
    # plt.title('morphed')
    # plt.figure()
    # plt.axis('off')
    # plt.imshow(final)
    # plt.title('final')
    # plt.show()


if __name__ == "__main__":
    cropImage()