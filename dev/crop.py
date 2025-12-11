import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt

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
    path_folder = "../data/horizontal_4m80_1/traitement_datas/images/"
    if len(sys.argv) > 1:
        number = int(sys.argv[1])
    else:
        number = 1
    image_path = f"{path_folder}image_{number}.png"
    print(f"Image : {image_path}")
    image = load_image(image_path)
    cropped_image = crop_image(image, 300, 77, 720, 600)
    cv.imshow("Cropped Image", cropped_image)
    cv.waitKey(0)
    cv.destroyAllWindows()