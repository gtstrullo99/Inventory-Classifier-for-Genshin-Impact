import cv2
import os
import numpy as np

import auto_crop as ac
import color_detector as cd
import item_identifier_orb as iio
import num_reader as nr


def recuperarNumero(listaNum):
    finalNum = 0
    for i in listaNum:
        if i != '\f' and i != '-' and i[0] != '.':
            finalNum *= 10
            finalNum += int(i[0])

    return finalNum


def main():
    pathTrain = './train'
    pathTest = './test/test9.jpg'

    ac.cropImage(pathTest)

    # Recibe el set de fotos (cada imagen es una celda con 1 solo item) y las clasifica individualmente según
    # varios criterios
    trainList = os.listdir(pathTrain)

    count = 0
    listaItems = {}
    for index, item in enumerate(trainList):
        print(index)
        curIm = cv2.imread(f'{pathTrain}/{item}')
        rarity = cd.colorDetection(curIm)

        clasNames, groundTruthImages = iio.searchClass(rarity)
        probArray = iio.featureDetection(curIm, rarity, clasNames, groundTruthImages)
        result = np.where(probArray == np.amax(probArray))

        listaNum = nr.identificarNumero(curIm)
        finalNum = recuperarNumero(listaNum)

        if clasNames[result[0][0]] not in listaItems:
            listaItems[clasNames[result[0][0]]] = finalNum
        else:
            count += 1


    print("Se han encontrado ", count, " objetos duplicado.")

    f = open("results.txt", "a")
    f.truncate(0)

    for i, k in zip(listaItems.values(), listaItems.keys()):
        k = " ".join(k.split()[1:])
        f.write(k + ": " + str(i) + '\n')

    f.close()

if __name__ == '__main__':
    main()