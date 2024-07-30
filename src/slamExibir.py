
import cv2
import numpy as np
import os
from utils import match_features, estimate_motion, draw_matches

def main():
    # Abrir vídeo
    cap = cv2.VideoCapture('video.mp4')

    # Verificar se o vídeo abriu corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo")
        return

    # Inicializar detector ORB
    orb = cv2.ORB_create()

    # Variáveis para armazenar a última imagem e pontos
    last_keypoints = None
    last_descriptors = None
    frame_idx = 0

    # Criar diretório para salvar frames processados
    if not os.path.exists('output'):
        os.makedirs('output')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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
            match_img = draw_matches(frame, keypoints, last_frame, last_keypoints, matches)

            # Salvar frame processado
            cv2.imwrite(f'output/match_{frame_idx:04d}.jpg', match_img)

        # Atualizar última imagem e características
        last_frame = frame.copy()
        last_keypoints = keypoints
        last_descriptors = descriptors
        frame_idx += 1

    cap.release()

if __name__ == '__main__':
    main()
