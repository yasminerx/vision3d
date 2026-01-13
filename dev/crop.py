import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt
import os

def load_image(path):
    """Load an image from the specified file path."""
    image = cv.imread(path)
    if image is None:
        raise FileNotFoundError(f"L'image n'exsite pas : {path}")
    return image

def crop_image(image, x, y, width, height):
    """Crop the image to the specified rectangle."""
    return image[y:y+height, x:x+width]





if __name__ == "__main__":
    # path_folder = "../data/horizontal_4m80_1/traitement_datas/images/"
    # output_folder = "../data/horizontal_4m80_1/traitement_datas/data_cropped/"
    path_folder = "../data/piscine_lent_1/traitement_datas/images/"
    output_folder = "../data/piscine_lent_1/traitement_datas/data_cropped/"
    
    # Créer le dossier s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)
    
    # Lister tous les fichiers PNG du dossier
    image_files = sorted([f for f in os.listdir(path_folder) if f.endswith('.png')])
    
    if not image_files:
        print(f"Aucune image trouvée dans {path_folder}")
        sys.exit(1)
    
    print(f"Traitement de {len(image_files)} images...")
    
    for image_file in image_files:
        image_path = os.path.join(path_folder, image_file)
        print(f"Traitement : {image_file}")
        
        try:
            image = load_image(image_path)
            #dimensions pour le horizontal_4m80_1
            # cropped_image = crop_image(image, 300, 77, 720, 600)
            #dimensions pour le piscine_lent_1
            cropped_image = crop_image(image, 220, 77, 940, 650) # x, y, width, height

            
            # Sauvegarder l'image croppée
            output_path = os.path.join(output_folder, image_file)
            cv.imwrite(output_path, cropped_image)
            print(f"  ✓ Sauvegardée : {output_path}")
        except Exception as e:
            print(f"  ✗ Erreur : {e}")
    
    print(f"\nTraitement terminé ! {len(image_files)} images traitées.")