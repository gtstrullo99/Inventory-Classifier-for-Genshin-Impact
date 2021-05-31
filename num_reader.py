#Alumno 1: Xiang Lin
#Alumno 2: Gerard Trullo

#Niu 1: 1465728
#Niu 2: 1494246


import pytesseract
import os
import cv2

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), 0)
        if img is not None:
            images.append(img)
    return images

def identificarNumero(img):
    try:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dim = img.shape
        crop_img = img[int((dim[0]/4))*3:dim[0], 0:dim[1]]

        #cv2.imshow('', img)
        #cv2.waitKey(0)

        #cv2.imshow('', crop_img)
        #cv2.waitKey(0)

        blur = cv2.GaussianBlur(crop_img, (5, 5), 0)
        #blur = cv2.medianBlur(crop_img, 3)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #cv2.imshow('', th3)
        #cv2.waitKey(0)

        canny = cv2.Canny(th3, 200, 300)

        #cv2.imshow('', canny)
        #cv2.waitKey(0)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        morphed = cv2.dilate(canny, kernel, iterations=3)

        #cv2.imshow('', morphed)
        #cv2.waitKey(0)


        cnts, hierarchy = cv2.findContours(morphed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        th4 = th3.copy()

        dimensions = crop_img.shape

        pad = 3

        # loop over the digit area candidates
        for c in cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            # if the contour is sufficiently large, it must be a digit

            if w <=50 and h >= 10 and h <= 25:

                (x, y, w, h) = cv2.boundingRect(c)
                th4 = cv2.rectangle(th4, (x, y), (x + w, y + h), (0, 255, 0), 2)

                #cv2.imshow('', th4)
                #cv2.waitKey(0)

                if y-pad >= 0 and y + h + pad <= dimensions[0] and x - pad >= 0 and x + w + pad <= dimensions[1]:
                    crop_num = crop_img[y - pad:y + h + pad, x - pad:x + w + pad]
                    ret4, th5 = cv2.threshold(crop_num, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    break


        #cv2.imshow('', th5)
        #cv2.waitKey(0)

        # percent by which the image is resized
        scale_percent = 2000

        # calculate the 50 percent of original dimensions
        width = int(th5.shape[1] * scale_percent / 100)
        height = int(th5.shape[0] * scale_percent / 100)

        # dsize
        dsize = (width, height)

        # resize image
        output = cv2.resize(th5, dsize)

        #cv2.imshow('', output)
        #cv2.waitKey(0)

        cnts, hierarchy = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

        dimensions = output.shape
        num = []
        for c in sorted_ctrs:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)
            # if the contour is sufficiently large, it must be a digit

            if 50 <= w <= 250 and h >= 200:
                #output = cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if y-30 >= 0 and y + h + 30 <= dimensions[0] and x - 30 >= 0 and x + w + 30 <= dimensions[1]:
                    final_output = output[y - 30:y + h + 30, x - 30:x + w + 30]

                    #cv2.imshow('', final_output)
                    #cv2.waitKey(0)

                    text = pytesseract.image_to_string(final_output, config='--psm 6 digits')
                    #print(text)

                    num.append(text)
    except:
        num = [[-1]]

    return num

if __name__ == "__main__":
    identificarNumero()