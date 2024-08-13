import cv2
import numpy as np
import os

import matplotlib.pyplot as plt

def show_image(img, title=''):
    """
    Exibe uma imagem usando matplotlib.
    :param img: Imagem para exibir.
    :param title: Título da imagem.
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def process_frame(frame, orb):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

def main(video_path):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    cap = cv2.VideoCapture(video_path)
    
    # Criar o diretório de saída se não existir
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prev_descriptors = None
    prev_keypoints = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, descriptors = process_frame(frame, orb)

        if prev_descriptors is not None:
            if descriptors is None or prev_descriptors is None:
                prev_descriptors = descriptors
                prev_keypoints = keypoints
                continue

            if descriptors.shape[1] != prev_descriptors.shape[1]:
                prev_descriptors = descriptors
                prev_keypoints = keypoints
                continue

            matches = bf.match(prev_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            img_matches = cv2.drawMatches(prev_frame, prev_keypoints, frame, keypoints, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Salvar a imagem na pasta output
            output_path = os.path.join(output_dir, f'matches_frame_{frame_count}.png')
            cv2.imwrite(output_path, img_matches)
            #print(f'Saved: {output_path}')
            frame_count += 1

        prev_frame = frame
        prev_keypoints = keypoints
        prev_descriptors = descriptors
    print(f'Saved: {output_path}')
    cap.release()

def show_images(output_dir='output', max_images=5):
    """
    Exibe as primeiras imagens salvas na pasta de saída usando matplotlib.
    :param output_dir: Diretório contendo as imagens a serem exibidas.
    :param max_images: Número máximo de imagens a serem exibidas.
    """
    images_shown = 0
    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith('.png'):
            if images_shown >= max_images:
                break
            img_path = os.path.join(output_dir, filename)
            img = cv2.imread(img_path)
            show_image(img, title=filename)  # Correto: passar a imagem e o título
            images_shown += 1

if __name__ == "__main__":
    video_path = 'video.mp4'  # Substitua pelo caminho do seu vídeo
    main(video_path)
    show_images()  # Exibir as imagens salvas
