

# ===========================================================================
# TP2 : INF7370 - Hiver 2022
#
# Indiquer votre nom ici
# Mohamad Hawchar : HAWM20039905
# ADEKOUDJO Ade-Dayo Nassir: ADEA04089904

#===========================================================================

# #===========================================================================
# Ce mod�le est un classifieur (un CNN) entrain� sur l'ensemble de donn�es MNIST afin de distinguer entre les images des chiffres 2 et 7.
# MNIST est une base de donn�es contenant des chiffres entre 0 et 9 �crits � la main en noire et blanc de taille 28x28 pixels
# Pour des fins d'illustration, nous avons pris seulement deux chiffres 2 et 7
#
# Donn�es:
# ------------------------------------------------
# entrainement : classe '2': 4 000 images | classe '7': images 4 000 images
# validation   : classe '2': 1 000 images | classe '7': images 1 000 images
# test         : classe '2': 1 000 images | classe '7': images 1 000 images 
# ------------------------------------------------

#>>> Ce code fonctionne sur MNIST. 
#>>> Vous devez donc intervenir sur ce code afin de l'adapter aux donn�es du TP. 
#>>> � cette fin rep�rer les section QUESTION et ins�rer votre code et modification � ces endroits

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# La libraire responsable du chargement des donn�es dans la m�moire

from keras.preprocessing.image import ImageDataGenerator

# Le Type de notre mod�le (s�quentiel)

from keras.models import Model
from keras.models import Sequential

# Le type d'optimisateur utilis� dans notre mod�le (RMSprop, adam, sgd, adaboost ...)
# L'optimisateur ajuste les poids de notre mod�le par descente du gradient
# Chaque optimisateur a ses propres param�tres
# Note: Il faut tester plusieurs et ajuster les param�tres afin d'avoir les meilleurs r�sultats

from tensorflow.keras.optimizers import Adam
#from keras.optimizers import Adam

# Les types des couches utlilis�es dans notre mod�le
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, UpSampling2D, Activation, Dropout, Flatten, Dense , LeakyReLU

# Des outils pour suivre et g�rer l'entrainement de notre mod�le
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

# Configuration du GPU
import tensorflow as tf
from keras import backend as K

# Affichage des graphes 
import matplotlib.pyplot as plt
import time
from datetime import timedelta
from keras.regularizers import l2
from keras.regularizers import l1
# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# ==========================================
# ================VARIABLES=================
# ==========================================

# ******************************************************
#                       QUESTION DU TP
# ******************************************************
# 1) Ajuster les variables suivantes selon votre probl�me:
# - mainDataPath
# - training_batch_size
# - validation_batch_size
# - image_scale
# - image_channels
# - images_color_mode
# - fit_batch_size
# - fit_epochs
# ******************************************************

# Le dossier principal qui contient les donn�es
mainDataPath = "donnees/"

# Le dossier contenant les images d'entrainement
trainPath = mainDataPath + "entrainement"

# Le dossier contenant les images de validation
validationPath = mainDataPath + "validation"

# Le dossier contenant les images de test
testPath = mainDataPath + "test"

# Le nom du fichier du mod�le � sauvegarder
modelsPath = "Model.hdf5"


# Le nombre d'images d'entrainement et de validation
# Il faut en premier lieu identifier les param�tres du CNN qui permettent d'arriver � des bons r�sultats. � cette fin, la d�marche g�n�rale consiste � utiliser une partie des donn�es d'entrainement et valider les r�sultats avec les donn�es de validation. Les param�tres du r�seaux (nombre de couches de convolutions, de pooling, nombre de filtres, etc) devrait etre ajust�s en cons�quence.  Ce processus devrait se r�p�ter jusqu'a l'obtention d'une configuration (architecture) satisfaisante. 
# Si on utilise l'ensemble de donn�es d'entrainement en entier, le processus va �tre long car on devrait ajuster les param�tres et reprendre le processus sur tout l'ensemble des donn�es d'entrainement.


training_batch_size = 21600  # 90% pour l'entrainement
validation_batch_size = 2400  # 10% pour la validation

# Configuration des  images 
image_scale = 120 # la taille des images
image_channels = 3  # le nombre de canaux de couleurs 3 pour RGB
images_color_mode = "rgb"  #  rgb pour les images en couleurs 
image_shape = (image_scale, image_scale, image_channels) # la forme des images d'entr�es, ce qui correspond � la couche d'entr�e du r�seau

# Configuration des param�tres d'entrainement
fit_batch_size = 32 # le nombre d'images entrain�es ensemble: un batch
fit_epochs = 100 # Le nombre d'�poques 

input_layer = Input(shape=image_shape)

def feature_extraction(input):
  
    #1
    x = Conv2D(128, (3, 3), padding='same')(input) 
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    #2
    x = Conv2D(128, (3, 3), padding='same')(x) 
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    #3
    x = Conv2D(128, (3, 3), padding='same')(x) 
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    #4   
    x = Conv2D(128, (3, 3))(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), padding='same')(x)  
    x = Dropout(0.3)(x)
    encoded = BatchNormalization()(x)
    
    return x

# Partie compl�tement connect�e (Fully Connected Layer)
def fully_connected(encoded):

    x = Flatten(input_shape=image_shape)(encoded)
    #x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(120, activity_regularizer = l2(0.001))(x) #
    #x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(6)(x)
    sortie = Activation('softmax')(x)
    return sortie


model = Model(input_layer, fully_connected(feature_extraction(input_layer)))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# ==========================================
# ==========CHARGEMENT DES IMAGES===========
# ==========================================

# training_data_generator: charge les donn�es d'entrainement en m�moire
# quand il charge les images, il les ajuste (change la taille, les dimensions, la direction ...) 
# al�atoirement afin de rendre le mod�le plus robuste � la position du sujet dans les images
# Note: On peut utiliser cette m�thode pour augmenter le nombre d'images d'entrainement (data augmentation)
training_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.1, # pour diviser les données en 90% entrainement et 10% validation
    # shear_range=0.1,
    # zoom_range=0.1,
    # rotation_range=20,
		#width_shift_range=0.2,
		#height_shift_range=0.2,
		#horizontal_flip=True,
		# fill_mode="nearest"
    )

# validation_data_generator: charge les donn�es de validation en memoire
validation_data_generator = ImageDataGenerator(rescale=1. / 255)

# training_generator: indique la m�thode de chargement des donn�es d'entrainement
training_generator = training_data_generator.flow_from_directory(
    trainPath, # Place des images d'entrainement
    color_mode=images_color_mode, # couleur des images
    target_size=(image_scale, image_scale),# taille des images
    batch_size=training_batch_size, # nombre d'images � entrainer (batch size)
    class_mode="categorical", # categorical parceque c'est un problem multi-class
    shuffle=True,# on "brasse" (shuffle) les donn�es -> pour pr�venir le surapprentissage
    subset='training') # pour indiquer que c'est le partie training, pour que la fonction le donne 90% des données

# validation_generator: indique la m�thode de chargement des donn�es de validation
validation_generator = training_data_generator.flow_from_directory(
    trainPath, # Place des images de validation
    color_mode=images_color_mode, # couleur des images
    target_size=(image_scale, image_scale),  # taille des images
    batch_size=validation_batch_size,  # nombre d'images � valider
    class_mode="categorical",  # categorical parceque c'est un problem multi-class
    shuffle=True, # on "brasse" (shuffle) les donn�es -> pour pr�venir le surapprentissage
    subset='validation' # pour indiquer que c'est le partie validation, pour que la fonction le donne 10% des données
    ) 

# On imprime l'indice de chaque classe (Keras numerote les classes selon l'ordre des dossiers des classes)
# Dans ce cas => [2: 0 et 7:1]
print(training_generator.class_indices)
print(validation_generator.class_indices)

# On charge les donn�es d'entrainement et de validation
# x_train: Les donn�es d'entrainement
# y_train: Les �tiquettes des donn�es d'entrainement
# x_val: Les donn�es de validation
# y_val: Les �tiquettes des donn�es de validation
(x_train, y_train) = training_generator.next()
(x_val, y_val) = validation_generator.next()
print("training shapes:",x_train.shape, " , ", y_train.shape)
print("validation shapes:",x_val.shape, " , ", y_val.shape)
# On Normalise les images en les divisant par la plus grande pixel dans les images (generalement c'est 255)
# Alors on aura des valeur entre 0 et 1, ceci stabilise l'entrainement
max_value = float(x_train.max())
x_train = x_train.astype('float32') / max_value
x_val = x_val.astype('float32') / max_value

# ==========================================
# ==============ENTRAINEMENT================
# ==========================================

# Savegarder le mod�le avec la meilleure validation accuracy ('val_accuracy') 
# Note: on sauvegarder le mod�le seulement quand la pr�cision de la validation s'am�liore
modelcheckpoint = ModelCheckpoint(filepath=modelsPath,
                                  monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')

# entrainement du mod�le
start = time.time()
classifier = model.fit(x_train, y_train,
                       epochs=fit_epochs, # nombre d'�poques
                       batch_size=fit_batch_size, # nombre d'images entrain�es ensemble
                       validation_data=(x_val, y_val), # donn�es de validation
                       verbose=1, # mets cette valeur � 0, si vous voulez ne pas afficher les d�tails d'entrainement
                       callbacks=[modelcheckpoint], # les fonctions � appeler � la fin de chaque �poque (dans ce cas modelcheckpoint: qui sauvegarde le mod�le)
                       shuffle=True)# shuffle les images 
execution_time = (time.time() - start)/60
# ==========================================
# ========AFFICHAGE DES RESULTATS===========
# ==========================================

# Plot accuracy over epochs (precision par �poque)
print(classifier.history.keys())
plt.plot(classifier.history['accuracy'])
plt.plot(classifier.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
fig = plt.gcf()
plt.show()

# ***********************************************
#                    QUESTION
# ***********************************************
#
# 4) Afficher le temps d'ex�cution
#
# ***********************************************
print("Total time: ", execution_time, " minutes")
# ***********************************************
#                    QUESTION
# ***********************************************
#
# 5) Ajouter la courbe de perte (loss curve)
#
# ***********************************************
plt.plot(classifier.history['loss'])
plt.plot(classifier.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

