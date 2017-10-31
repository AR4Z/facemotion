import cv2
import numpy as np
from PIL import Image


def image_as_nparray(image):
    """
    convierte imagenes de PIL's a matrices.
    :param image: una imagen de PIL.
    :return: la imagen convertida en una matriz.
    """
    return np.asarray(image)


def nparray_as_image(nparray, mode='RGB'):
    """
    convierte una matriz de una imagen a una imagen de PIL.
    :param nparray: matriz de una imagen.
    :param mode: modo de conversion pr defecto es RGB
    :return: la matriz convertida en imagen.
    """
    return Image.fromarray(np.asarray(np.clip(nparray, 0, 255), dtype='uint8'), mode)


def load_image(source_path):
    """
    carga la imagen cnvertida a escala de grises
    :param source_path: ruta de la imagen.
    :return: imagen cargada convertida a escala de grises.
    """
    source_image = cv2.imread(source_path)
    return cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)


def draw_with_alpha(source_image, image_to_draw, coordinates):
    """
    dibuja una imagen sobre otra imagen.
    :param source_image: imagen para dibujar encima.
    :param image_to_draw: donde se va a dibujar la imagen.
    :param coordinates: coordenada de donde se va a dibujar la imagen su ancho y alto.
    """
    x, y, w, h = coordinates
    image_to_draw = image_to_draw.resize((h, w), Image.ANTIALIAS)
    image_array = image_as_nparray(image_to_draw)
    for c in range(0, 3):
        source_image[y:y + h, x:x + w, c] = image_array[:, :, c] * (image_array[:, :, 3] / 255.0) \
                                            + source_image[y:y + h, x:x + w, c] * (1.0 - image_array[:, :, 3] / 255.0)
