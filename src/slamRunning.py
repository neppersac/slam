import cv2
import numpy as np

def process_frame(frame, orb, bf):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

def main(video_path):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    cap = cv2.VideoCapture(video_path)

    print("Ler o arquivo de video")

    prev_descriptors = None
    prev_keypoints = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, descriptors = process_frame(frame, orb, bf)

        if prev_descriptors is not None:
            if descriptors is None or prev_descriptors is None:
                # Não há descritores em um ou ambos os frames
                prev_descriptors = descriptors
                prev_keypoints = keypoints
                continue

            if descriptors.shape[1] != prev_descriptors.shape[1]:
                # Incompatibilidade no número de colunas dos descritores
                prev_descriptors = descriptors
                prev_keypoints = keypoints
                continue

            # Correspondência de características entre frames
            matches = bf.match(prev_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            img_matches = cv2.drawMatches(prev_frame, prev_keypoints, frame, keypoints, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(f'matches_frame_{frame_count}.png', img_matches)
            frame_count += 1

        prev_frame = frame
        prev_keypoints = keypoints
        prev_descriptors = descriptors

    cap.release()

if __name__ == "__main__":
    video_path = 'video.mp4'  # Substitua pelo caminho do seu vídeo
    main(video_path)
