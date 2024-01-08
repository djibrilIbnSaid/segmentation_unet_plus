# segmentation_unet_plus

Projet (Introdution à l'apprentissage profond) de segmentation d'images médicales par réseaux de neurones convolutifs en utilisant UNet++.

## Prérequis
Python 3.* et les librairies trouvables dans le fichier `requirements.txt`.

## Installation
Pour installer les librairies nécessaires, il suffit de lancer la commande suivante dans le terminal :
```
pip install -r requirements.txt
```

## Explication des fichiers et dossiers
- `analyse.ipynb` : Notebook Jupyter servant à analyser les résultats obtenus a partir des entrainés sur colab.

- `save_model.py` : Script python servant à instancifier un modèle UNet++ a partir du projet de base. Ce modèle est utilisé sur colab.

- `train_on_colab.ipynb` : Notebook Jupyter servant à entrainer le modèle UNet++ sur colab.

- `tutils.py` : Script python contenant les fonctions utilitaires pour le projet.

- `kerass` : Dossier contenant le projet de base.

- `docs` : Dossier contenant les documents de présentation du projet.

##### NB: Dans la documentation github c'est mentionné d'utiliser le fichier `DSB2018_application.py` pour l'entrainement du modèle, mais ce fichier n'est pas présent dans le dossier `kerass` du projet de base. C'est pour cela que nous avons utilisé le fichier `train_on_colab.py` et utiliser leur option pour un entraiment avec mes propre données pour l'entrainement du modèle. En ce qui concerne la base `BRATS 2013` je n'ai pas reçu la base a temps parce qu'une personne confirme ma demande.

##### J'ai modifié le dossier `keras` du projet en `kerass` pour eviter les conflits pour les imports.