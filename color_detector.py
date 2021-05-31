import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

'''
    Librerías necesarias:
        · cv2: pip install opencv-python
        · matplotlib: pip install matplotlib
        · numpy: pip install numpy
        · os: para recuperar nombres de archivos
'''


'''
    Hará una detección de color para identificar:
        - Naranja: legendario (0)
        - Morado: épico (1)
        - Azul: raro (2)
        - Verde: poco común (3)
        - Gris: común (4)
    El color que sea más prometedor, será el resultado que devolverá
    
    Input: 
        - image: array de valores en 3D que representará la imagen
    Output:
        - rarity: 1 de las 5 clases a las que será más probable que pertenezca
'''
def colorDetection(image):
    # Se inicializa la rareza inicial del item y el contador de coincidencia máxima
    rarity = -1
    maxCount = -1

    # Creamos la array de rarezas
    rarityArray = ['legendary', 'epic', 'rare', 'uncommon', 'common']

    # Transformamos la imagen a HSV, ya que así trabaja mejor cv2
    imHSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Se establecen los diferentes lowers y upper limits según la rareza de los objetos:
    # Legendary = Naranja, Epic = Morado, Rare = Azul, Uncommon = Verde, Common = Gris
    rarityLower = [
         [100, 210, 140], [120, 110, 110], [0, 65, 130], [40, 40, 120], [114, 60, 100]
    ]
    rarityUpper = [
         [115, 255, 255], [140, 180, 250], [20, 130, 180], [90, 130, 190], [130, 110, 170]
    ]
    #plt.figure(figsize=(10, 8))
    #plt.imshow(imHSV)

    for lower, upper, index in zip (rarityLower, rarityUpper, rarityArray):
        lowerLimit = np.array(lower)
        upperLimit = np.array(upper)

        # Se crea una máscara para ver las zonas que ha elegido el algoritmo con los límites
        maskHSV = cv2.inRange(imHSV, lowerLimit, upperLimit)
        #plt.figure(figsize=(10, 8))
        #plt.imshow(maskHSV)

        # Se discrimina lo que no haya elegido la máscara
        res = cv2.bitwise_and(imHSV, imHSV, mask=maskHSV)
        #plt.figure(figsize=(10, 8))
        #plt.imshow(res)

        counter = cv2.countNonZero(res[:, :, 0])
        if counter > maxCount:
            maxCount = counter
            rarity = index

    #plt.close('all')

    return rarity
    
'''
    La función main debería recibir una o varias imágenes de una pantalla de un PC hecha con el móvil (movimiento, 
    desenfoque, diferentes ángulos...) con el inventario de mejora del juego Genshin Impact y el programa debería 
    ser capaz de clasificar los items por nombre, rareza y cantidad.
    Luego, esta información sería impresa en un archivo .txt en el orden que se encontraba en la imagen para poder
    subirla más rápidamente a una web que analiza los items para optimizar una ruta de min-maxing de recursos.
    
    Dicha web precisa que se copien todos los datos del inventario a mano, por lo que es un trabajo bastante
    laborioso. https://seelie.inmagi.com/inventory
'''
def main():
    # Se declara el path del train
    pathTrain = 'train'

    # Recibe el set de fotos (cada imagen es una celda con 1 solo item) y las clasifica individualmente según
    # varios criterios
    trainList = os.listdir(pathTrain)
    for item in trainList:
        curIm = cv2.imread(f'{pathTrain}/{item}')
        rarity = colorDetection(curIm)

if __name__ == '__main__':
    main()

