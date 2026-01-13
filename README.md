# Vision 3d - Reconstitution du mur à partir d'un flux vidéo

Commencer par installer les dépendances (voir requirements.txt).
Les données brutes sont disponibles dans data.
### 1. Traiter les images
On crop les images pour avoir uniquement le mur.
```bash
cd dev
python crop.py
# Génère: ../data/horizontal_4m80_1/traitement_datas/data_cropped/
```

### 2. Lancer la reconstruction 3D
```bash
cd sfm-master
python3 run.py -i data/horizontal_4m80_cropped/ 

# Options
-s, --scale          Échelle (default=100). Ex: -s 50 pour 50%
-n, --num_images     Nombre d'images à traiter (default=toutes)
-p, --plot           Afficher la reconstruction en temps réel
-o, --outfile        Fichier de sortie (default=results/out.npz)
--close-loop         Boucler la trajectoire (pour 360°, désactivé par défaut)
# Sortie: results/out.npz
```

### 3. Visualiser le résultat
```bash
python3 plot.py results/out.npz
```
