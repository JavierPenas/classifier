
import Loader as ld
import tflearn
import os
import SimpleNetwork as sn
import keras
import tensorflow as tf

# CREAMOS DIRECTORIO PARA EL MODELO
MODEL_PATH = "/home/javier/DevData/model"
MODEL_NAME = "classificator.model"
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH, mode=0o755)

if __name__ == '__main__':

    #CARGAMOS LAS COLECCIONES DE IMAGENES
    loader = ld.Loader("/home/javier/DevData/")
    (X, Y), (X_test, Y_test) = loader.loadData()

    # TODO IDENTIFICAR QUE HACEN ESTAS FUNCIONES
    Y = tflearn.data_utils.to_categorical(Y, 2)
    Y_test = tflearn.data_utils.to_categorical(Y_test, 2)

    # ENTRENAMIENTO RRNN NORMAL
    print("[INFO] CREATING MODEL WITH CLASSICAL NEURAL NETWORK")
    model_name_NN = 'nn_' + MODEL_NAME
    model_NN = tflearn.DNN(sn.nn(), checkpoint_path='model_NN',
                           max_checkpoints=10, tensorboard_verbose=3)

    print("[INFO] TRAINING CLASSICAL NEURAL NETWORK")
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(128, 128)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X, Y, epochs=5)
    test_loss, test_acc = model.evaluate(X_test,Y_test)
    print("[INFO] TEST ACCURACY: ", test_acc)
