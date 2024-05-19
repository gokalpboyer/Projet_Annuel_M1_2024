# Dans votre fichier views.py

from django.shortcuts import render
from PIL import Image

import ctypes
import time
import numpy as np
import os

import matplotlib.pyplot as plt

def process_photo_view(request):
    if request.method == 'POST' and request.FILES['photo']:
        photo_file = request.FILES['photo']

        # Vérifiez si le fichier est bien une image JPG
        if photo_file.name.endswith('.jpg') or photo_file.name.endswith('.jpeg'):
            # Ouvrir l'image avec Pillow
            image = Image.open(photo_file)

            # Traitez l'image comme vous le souhaitez
            # Par exemple, redimensionnez l'image et sauvegardez-la dans un autre emplacement
            resized_image = image.resize((300, 200))
            #resized_image.save(r'C:\Users\desan\Documents\ESGI\PA\pythonProject2\dataset\predict\image_resized.jpg')
            prediction = predic(photo_file)
            nom_classe = "rien"
            if prediction == 1:
                nom_classe = "football"
            elif prediction == 2:
                nom_classe = "basket"
            elif prediction == 3:
                nom_classe = "tennis"

            return render(request, 'result.html', {'message': nom_classe})
        else:
            return render(request, 'error.html', {'message': 'Le fichier doit être une image JPG.'})
    else:
        return render(request, 'upload.html')




# Convertie l'image en matrice
def get_matrice_image(img):
    img_mat = np.array(img)
    return img_mat


    # Retourne la liste d'image d'un dossier
def get_img_list(directory, largeur, hauteur):
    img_dir = directory
    img_list = []
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        img_resize = img.resize((largeur, hauteur))
        img_black_and_white = img_resize.convert("L")
        img_list.append(img_black_and_white)
    return img_list


def get_list_img_flatten(path_data, largeur, hauteur):
    img_list = get_img_list(path_data, largeur, hauteur)
    img_flatten_list = []
    for img in img_list:
        img_mat = get_matrice_image(img)
        img_flatten = img_mat.flatten()
        img_flatten_list.append(img_flatten)
    return img_flatten_list









def predict_class_img(img, img_size, path_file_model_football, path_file_model_basket,
                      path_file_model_tennis, my_lib):
    func = my_lib.predict_class
    """ctype_img = img.ctypes.data_as(ctypes.POINTER(ctypes.c_int))"""
    ctype_img = (ctypes.c_int * len(img))(*img)
    """for i in range(len(img)):
        print("PIXELS", i, "=", ctype_img[i])"""
    func.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p,
                     ctypes.c_char_p]
    func.restype = ctypes.c_int
    list_train = func(ctype_img, img_size, path_file_model_football.encode(),
                      path_file_model_basket.encode(), path_file_model_tennis.encode())
    return list_train






def predic(img):
    largeur_img = 50
    hauteur_img = 50

    # IMPORTANT mettre les chemins absolus des fichiers
    path_file_model_football = "D:/PA-DJANGO/testDjango/football_model.txt"
    path_file_model_basket = "D:/PA-DJANGO/testDjango/basket_model.txt"
    path_file_model_tennis = "D:/PA-DJANGO/testDjango/tennis_model.txt"

    new_image = Image.open(img).resize(
        (largeur_img, hauteur_img)).convert("L")
    new_img_flatten = get_matrice_image(new_image).flatten()

    ######################################### On intéroge le modèle ###########################################
    prediction = predict_class_img(new_img_flatten, largeur_img * hauteur_img, path_file_model_football,
                                   path_file_model_basket, path_file_model_tennis, my_lib)
    return prediction


