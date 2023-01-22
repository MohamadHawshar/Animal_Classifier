
# ===========================================================================
# TP2 : INF7370 - Hiver 2022
#
# Indiquer votre nom ici
# Mohamad Hawchar : HAWM20039905
# ADEKOUDJO Ade-Dayo Nassir: ADEA04089904
#===========================================================================


#===========================================================================
# Dans ce script, on �value le mod�le entrain� dans 1_Modele.py
# On charge le mod�le en m�moire; on charge les images; et puis on applique le mod�le sur les images afin de pr�dire les classes



# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# La libraire responsable du chargement des donn�es dans la m�moire
from keras.preprocessing.image import ImageDataGenerator

# Affichage des graphes
import matplotlib.pyplot as plt

# La librairie numpy 
import numpy as np

# Configuration du GPU
import tensorflow as tf
from keras import backend as K

# Utilis� pour le calcul des m�triques de validation
from sklearn.metrics import confusion_matrix, roc_curve , auc, multilabel_confusion_matrix

# Utlilis� pour charger le mod�le
from keras.models import load_model
from keras import Model


# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# ==========================================
# ==================MOD�LE==================
# ==========================================

#Chargement du mod�le sauvegard� dans la section 1 via 1_Modele.py
model_path = "Model.hdf5"
Classifier : Model = load_model(model_path)

# ==========================================
# ================VARIABLES=================
# ==========================================

# ******************************************************
#                       QUESTIONS
# ******************************************************>
# 1) A ajuster les variables suivantes selon votre probl�me:
# - mainDataPath
# - number_images
# - number_images_class_0
# - number_images_class_1
# - image_scale
# - images_color_mode
# - images_color_mode
# ******************************************************


# L'emplacement des images de test
mainDataPath = "donnees/"
testPath = mainDataPath + "test"

# Le nombre des images de test � �valuer
number_images = 6000 # 1000 images pour la classe du chiffre 2 et 1000 pour la classe du chiffre 7
number_images_class_0 = 1000
number_images_class_1 = 1000
number_images_per_class = 1000

# La taille des images � classer
image_scale = 120

# La couleur des images � classer
images_color_mode = "rgb"  # grayscale or rgb

# ==========================================
# =========CHARGEMENT DES IMAGES============
# ==========================================

# Chargement des images de test
test_data_generator = ImageDataGenerator(rescale=1. / 255,
                                         )

test_itr = test_data_generator.flow_from_directory(
    testPath,# place des images
    target_size=(image_scale, image_scale), # taille des images
    class_mode="categorical",# Type de classification
    shuffle=False,# pas besoin de les boulverser
    batch_size=1,# on classe les images une � la fois
    color_mode=images_color_mode)# couleur des images

(x, y_true) = test_itr.next()

# getting the paths of the files
count = 0
list_of_generated_files = []
for filepath in test_itr.filepaths:
  count+=1
  list_of_generated_files.append(filepath)
  
print("count: " ,count , " shape of list: ", len(list_of_generated_files))




# Normalize Data
max_value = float(x.max())
x = x.astype('float32') / max_value

# ==========================================
# ===============�VALUATION=================
# ==========================================

# Les classes correctes des images (1000 pour chaque classe) -- the ground truth
y_true = np.array([0] * number_images_per_class + 
                  [1] * number_images_per_class + 
                  [2] * number_images_per_class + 
                  [3] * number_images_per_class + 
                  [4] * number_images_per_class + 
                  [5] * number_images_per_class )

# evaluation du mod�le
test_eval = Classifier.evaluate_generator(test_itr, verbose=1)

# Affichage des valeurs de perte et de precision
print('>Test loss (Erreur):', test_eval[0])
print('>Test pr�cision:', test_eval[1])

# Pr�diction des classes des images de test
predicted_classes = Classifier.predict_generator(test_itr, verbose=1)
#print("predicted class 0 shape:",predicted_classes.shape,"\n \t values: ",predicted_classes)
predicted_classes_perc = np.round(predicted_classes.copy(), 4)
#print("predicted class 1 shape:",predicted_classes.shape,"\n \t values: ",predicted_classes)
predicted_classes = np.round(predicted_classes) # on arrondie le output
#print("predicted class 2 shape:",predicted_classes.shape,"\n \t values: ",predicted_classes)
# 0 => classe 2
# 1 => classe 7

# Cette list contient les images bien class�es
correct = []
for i in range(0, len(predicted_classes) - 1):
    if np.where(predicted_classes[i] == 1)  == y_true[i]:
        correct.append(i)

# Nombre d'images bien class�es
print("> %d  �tiquettes bien class�es" % len(correct))

# Cette list contient les images mal class�es
incorrect = []
for i in range(0, len(predicted_classes) - 1):
    if np.where(predicted_classes[i] == 1) != y_true[i]:
        incorrect.append(i)

# Nombre d'images mal class�es
print("> %d �tiquettes mal class�es" % len(incorrect))

# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 2) Afficher la matrice de confusion
# 3) Afficher la courbe ROC
# 4) Extraire 5 images de chaque cat�gorie
#
# ***********************************************
# 1 Confusion Matrix
import scikitplot as skplt

y_pred = np.argmax(predicted_classes,axis=1)
labels=["elephant", "girafe", "leopard", "rhino","tigre","zebre"]
# switiching the indexs with the labels
y_pred = np.array([labels[i] for i in y_pred])
y_true = np.array([labels[i] for i in y_true])


skplt.metrics.plot_confusion_matrix(
    y_true, 
    y_pred, 
    #normalize=True,
    cmap = "Purples",
    figsize=(8,8)    
)
#2 5 images de chaque cat�gorie

import pandas as pd
from glob import glob
import os

path = '/content/donnees/test'
df = pd.DataFrame(columns=['FileName','Category'])
predictions_df = pd.DataFrame(y_pred,columns=['Prediction'])
prediction_paths_df = pd.DataFrame(list_of_generated_files,columns = ['Prediction-Path'])
predictions_paths = pd.DataFrame(y_pred,columns=['Prediction'])  
test_file_paths = []
for label in labels:
  test_file_paths = sorted(glob(os.path.join(path, label, "*.png")))
  for p in test_file_paths:
    df = df.append({'FileName':p, 'Category':label}, ignore_index=True)
  
print(len(test_file_paths)) 
predictions_df = pd.concat([predictions_df,prediction_paths_df],axis=1)
df = pd.concat([df,predictions_df],axis=1)
display(df)
#display(predictions_df)

from typing_extensions import final
from PIL import Image

rows,cols = 6,6
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15,15))
plt.subplots_adjust(wspace=0.1, hspace=0.1)
j=0
i=0
k = 0
for i in range(0,6):
  for j in range(0,6):

    l1 = labels[i]
    l2 = labels[j]

    if i == 0:    
      axes[i//cols, j%cols].set_title(l2)
    if j == 0:
      axes[i, j].set_ylabel(l1)   
    try:
      path = df[(df['Category'] == l1) & (df['Prediction'] == l2)].head(1)
      path = path['FileName'].iloc[0]
      img = Image.open(path)
       
      axes[k//cols, k%cols].imshow(img.resize((120,120)))
      axes[k//cols, k%cols].set_xticklabels([])
      axes[k//cols, k%cols].set_yticklabels([])
      k+=1
    except :
      # print the case where there is no animal x calssified as animal y. in our case it prints (leopard /rhino, leopard /zebre)
      axes[k//cols, k%cols].imshow(np.zeros((120,120)))
      axes[k//cols, k%cols].set_xticklabels([])
      axes[k//cols, k%cols].set_yticklabels([])
      print(f"{l1} /{l2}")
      k+=1
  # print("---------------------")
    
