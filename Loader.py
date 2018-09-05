import numpy as np
import os
from os.path import isfile,join
from scipy.misc import imresize
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Loader:

    #INICIALIZAMOS LOS ARRAYS QUE CONTENDRAN LAS ETIQUETAS Y LAS IMAGENES
    Labels = []
    Images = []
    #SETEAMOS EL PATH POR DEFECTO DE DONDE SE CARGAN LOS DATASET
    myPath = "/home/javier/DevData"
    CLASS_A_FOLDER = "/cat"
    CLASS_B_FOLDER = "/dog"
    # PARAMETROS PARA DIVIDIR LAS COLECCIONES DE IMAGENES EN EL CONJUNTO ENTRENAMIENTO Y TEST
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    def __init__(self,path):
        self.myPath = path

    def setNewPath(self,newPath):
        self.myPath = newPath

    def read_img(self,path,dshape):
        image = Image.open(path)
        image = image.resize((128, 128), Image.ANTIALIAS)
        gray = image.convert('L')
        bw = gray.point(lambda x: 0 if x<128 else 255, '1')
        new_image = bw
        # new_image = imresize(image, tuple(dshape))
        return new_image

    def loadData(self):

        #LEEMOS DE FICHERO LAS IMAGENES
        print("[INFO] LOADING TRAINING AND TESTING DATASET FROM:" + self.myPath)
        #TODO MEJORA
        classA = [self.myPath + self.CLASS_A_FOLDER + "/" + fPath
                  for fPath in os.listdir(self.myPath+self.CLASS_A_FOLDER) if
                  isfile(join(self.myPath+self.CLASS_A_FOLDER,fPath)) and fPath.endswith(".jpg")]
        classB = [self.myPath + self.CLASS_B_FOLDER + "/" + fPath
                  for fPath in os.listdir(self.myPath+self.CLASS_B_FOLDER) if
                  isfile(join(self.myPath+self.CLASS_B_FOLDER,fPath)) and fPath.endswith(".jpg")]

        for f in classA:
            label = 0
            image = self.read_img(f,[128,128,3])
            self.Labels.append(label)
            self.Images.append(image)
        for f in classB:
            label = 1
            image = self.read_img(f, [128, 128, 3])
            self.Labels.append(label)
            self.Images.append(image)

        print("[INFO] SPLIT THE TRAINING AND TESTING DATASETS")
        X, X_test, y, y_test = train_test_split(self.Images, self.Labels,
                                                                             test_size=self.TEST_SIZE,
                                                                             random_state=self.RANDOM_STATE)
        print("[INFO] IMAGE DATASET SUCCESSFULLY LOADED AND SPLITTED")
        print("[DEBUG] TRAIN SET SIZE: "+ len(X).__str__())
        print("[DEBUG] TEST SET SIZE: " + len(y).__str__())

        return (X, y), (X_test, y_test)


if __name__ == '__main__':

    loader = Loader("/home/javier/DevData")
    (X, Y), (X_test, Y_test) = loader.loadData()
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid('off')
        plt.imshow(X[i], cmap=plt.cm.binary)
    plt.show()
