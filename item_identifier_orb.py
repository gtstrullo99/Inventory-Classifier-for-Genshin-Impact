import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

'''
    Detecta los descriptores de las imagenes de groundtruth y las guarda en una lista.

    Input: 
        - images: set de imágenes de Groundtruth
        - sift: clasificador
    Output:
        - desList: lista con todos los descriptores de las clases candidatas
'''


'''
Se ha intentado hacer comparación de imágenes con MSE, pero los resultados no eran óptimos

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    print("hello")
    aWidth, aHeight = imageA.size
    bWidth, bHeight = imageB.size

    if(aWidth * aHeight > bWidth * bHeight):
        im = imageA.resize((bWidth, bHeight))
        im.save("tempA.jpg")
        imageB.save("tempB.jpg")
    else:
        im = imageB.resize((aWidth, aHeight))
        im.save("tempB.jpg")
        imageA.save("tempA.jpg")

    imageA = cv2.imread("tempA.jpg")
    imageB = cv2.imread("tempB.jpg")

    imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    m = mse(imageA, imageB)
    s = ssim(imageA, imageB, multichannel=True)

    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("off")

    # show the images
    plt.show()


def differenceDetector(colorFeature, rarity):
    clasNames, groundTruthImages = searchClass(colorFeature, rarity, 1)

    # read images
    img2 = Image.open('train/Varunada Lazurite Chunk 1.jpeg')

    for index, image in enumerate(groundTruthImages):
        compare_images(img2, image, 'foto ' + str(index))
'''


def findTestDescriptors(images, orb):
    desList = []
    kpList = []
    for image in images:
        kp, des = orb.detectAndCompute(image, None)
        desList.append(des)
        kpList.append(kp)

    return desList, kpList


def searchClass(rarity):
    # Se declara el path del groundtruth
    pathGroundtruth = './groundtruth'
    # groundtruthFolders = ['legendary', 'epic', 'rare', 'uncommon', 'common']

    # Se guardan en dos arrays las imagenes en formato RGB y los nombres de las imagenes (sin extensión)
    classNames = []
    groundtruthImages = []

    # Se obtienen todos los nombres de la carpeta de Groundtruth.
    # (Se tendrá que cambiar para que mire dentro de las subcarpetas: legendario, epico, raro...)

    classList = os.listdir(f'{pathGroundtruth}/{rarity}')

    for item in classList:
        # Carga las imagenes en formato normal (BGR) y las pasa a (RGB).
        curIm = cv2.imread(f'{pathGroundtruth}/{rarity}/{item}', 1)
        curIm = cv2.cvtColor(curIm, cv2.COLOR_BGR2RGB)

        groundtruthImages.append(curIm)
        classNames.append(os.path.splitext(item)[0])

    return classNames, groundtruthImages


'''
    Recibe las imagenes de Groundtruth (se tendrá que actualizar para que solo recibe un subset de imagenes, por 
    ejemplo, el subset de legendarias, épicas...) y la función las comparará con la imagen que tiene que clasificar.

    Input: 
        - testImages: set de imagenes en la que puede acabar clasificada la imagen
        - image: imagen que se ha de clasificar
        todo - rarity: rareza del item devuelto por colorDetection
    Output:
        - resultado: clasificación del item
'''


def featureDetection(image, rarity, clasNames, groundTruthImages):
    # Se declara un orb que servirá como el clasificador
    orb = cv2.ORB_create()

    # La función extrae los descriptores de la lista de imagenes y los guarda en una lista
    descriptorList, keyPointList = findTestDescriptors(groundTruthImages, orb)

    img_denoised = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
    # Detecta los descriptores de la imagen a clasificar
    kp2, des2 = orb.detectAndCompute(img_denoised, None)

    # Inicializa un Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matchList = []

    # Intenta buscar las coincidencias...
    try:
        # ... por cada descriptor de las posibles clases
        for i, des in enumerate(descriptorList):
            matches = bf.match(des, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            #img3 = cv2.drawMatches(groundTruthImages[i], keyPointList[i], image, kp2, matches[:50], image, flags=2)
            #plt.imshow(img3), plt.show()

            # # Si está por encima de cierto umbral de coincidencia, guarda dicho valor
            good = [i for i in matches if i.distance < 1100]

            # Al final tenemos una array de tamaño N (siendo N el número de clases candidatas) donde cada posición
            # indica la cantidad de features coincidentes. Por lo tanto, cuanto más grande el número, más probable
            # que sea de dicha clase.
            matchList.append(len(good))
    except:
        pass

    return np.array(matchList)


def main():
    # Se declara el path del train
    pathTrain = 'train'

    rarity = 'epic'

    # Se establece si se hará la feature detection en RGB (1) o GRIS (0)
    colorFeature = 1

    # Recibe el set de fotos (cada imagen es una celda con 1 solo item) y las clasifica individualmente según
    # varios criterios
    trainList = os.listdir(pathTrain)
    for item in trainList:
        curIm = cv2.imread(f'{pathTrain}/{item}')
        curIm = cv2.cvtColor(curIm, cv2.COLOR_BGR2RGB)
        probaArray = featureDetection(curIm, rarity)
        result = np.where(probaArray == np.amax(probaArray))


if __name__ == '__main__':
    main()
