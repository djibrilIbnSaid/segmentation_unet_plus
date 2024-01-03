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