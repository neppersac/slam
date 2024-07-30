import cv2
import os
from utils import load_images, match_features, estimate_motion, draw_matches

def main():
    # Carregar sequencia de imagens
    images = load_images('images')
    print(images)

    # Inicializar detector ORB
    orb = cv2.ORB_create()

    # Variáveis para armazenar última imagem e pontos
    last_keypoints = None
    last_descriptors = None

    for i, frame in enumerate(images):
        # Converter para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar características e calcular descritores
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if last_descriptors is not None:
            # Correspondência de características
            matches = match_features(descriptors, last_descriptors)

            # Estimar movimento
            R, t = estimate_motion(matches, keypoints, last_keypoints)

            # Desenhar correspondências
            match_img = draw_matches(frame, keypoints, images[i-1], last_keypoints, matches)
            cv2.imshow('Matches', match_img)
            cv2.waitKey(0)

        # Atualizar última imagem e características
        last_keypoints = keypoints
        last_descriptors = descriptors

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
