import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


DATA_DIR = 'data/plantDoc_leaf_disease' # Chemin vers le dossier des données

def get_image_folder(train=True, augmented=False):
    """ Cette fonction prend en entrée deux booléens et retourne les chemins vers le dossier des images ou le masque.

    Args:
        train (bool, optionnel): True pour les images d'entrenement sinon False pour les images de teste . Par defaut True.
        mask (bool, optionnel): True pour les images masquées sinon False pour les images non masquées . Par defaut False.
        augmented (bool, optionnel): True pour les images augmentees sinon False pour les images de base . Par defaut False.
    
    Returns:
        str: Le chemin vers le dossier des images.
        str: Le chemin vers le dossier des masques.
    """
    if train:
        if augmented:
            return (os.path.join(DATA_DIR, 'aug_data', 'train', 'images'), os.path.join(DATA_DIR, 'aug_data', 'train', 'masks'))
        else:
            return (os.path.join(DATA_DIR, 'data', 'train', 'images'), os.path.join(DATA_DIR, 'data', 'train', 'masks'))
    else:
        if augmented:
            return (os.path.join(DATA_DIR, 'aug_data', 'test', 'images'), os.path.join(DATA_DIR, 'aug_data', 'test', 'masks'))
        else:
            return (os.path.join(DATA_DIR, 'data', 'test', 'images'), os.path.join(DATA_DIR, 'data', 'test', 'masks'))


def load_data(images_folder, masks_folder, target_size=(256, 256)):
    """ Cette fonction prend en entrée deux chemins vers des dossiers, la taille des images et retourne deux listes contenant les images et les masques.

    Args:
        images_folder (str): Le chemin vers le dossier des images.
        masks_folder (str): Le chemin vers le dossier des masques.
    
    Returns:
        list: La liste des images.
        list: La liste des masques.
    """
    images = []
    masks = []
    for image_name in os.listdir(images_folder):
        if image_name.endswith('.jpg'):
            image_path = os.path.join(images_folder, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, target_size)
            images.append(image)
            
            mask_filename = image_name.split('.')[0] + '.png'
            mask_path = os.path.join(masks_folder, mask_filename)
            mask = cv2.imread(mask_path)
            mask = cv2.resize(mask, target_size)
            masks.append(mask)     
    print(f"{len(images)} images chargées")
    return np.array(images), np.array(masks)    