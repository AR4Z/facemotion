import cv2

# modelo para detectar rostros con opencv y cascadas
faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')


def find_faces(image):
    # lista con las coordenadas de los rostros encontrados
    faces_coordinates = _locate_faces(image)

    # rostros cortados
    cutted_faces = [image[y:y + h, x:x + w] for (x, y, w, h) in faces_coordinates]

    # convierte cada rostro encontrado para poder trabajar en a deteccion de emociones
    normalized_faces = [_normalize_face(face) for face in cutted_faces]
    # lista de tuplas con ada uno de los rostros  su respectiva coordenada
    return zip(normalized_faces, faces_coordinates)


def _normalize_face(face):
    # convierte cada rostro encontrado para poder trabajar en a deteccion de emociones
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # para usar el fisherFace todas las imagenes deben tener igual tamano
    face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
    # devuelve el nuevo rostro
    return face


def _locate_faces(image):
    # detecta rostros en una imagen y devuelve sus coordenadas y dimensiones
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize=(70, 70)
    )

    return faces
