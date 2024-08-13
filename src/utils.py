import cv2
import numpy as np

def match_features(img1, img2):
    """
    Detecta, descreve e corresponde características entre duas imagens usando SIFT e FLANN.

    :param img1: A primeira imagem.
    :param img2: A segunda imagem.
    :return: keypoints1, keypoints2, matches - keypoints e matches encontrados.
    """
    # Verificar o número de canais e converter para escala de cinza, se necessário
    if len(img1.shape) == 3:  # Imagem colorida (3 canais)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:  # Imagem já em escala de cinza (1 canal)
        gray1 = img1

    if len(img2.shape) == 3:  # Imagem colorida (3 canais)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:  # Imagem já em escala de cinza (1 canal)
        gray2 = img2

    # Detecta e descreve as características usando SIFT
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Configura o FLANN para correspondência de características
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Encontrar correspondências usando o método KNN
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Aplicar a razão de Lowe para filtrar correspondências
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return keypoints1, keypoints2, good_matches

def estimate_motion(keypoints1, keypoints2, matches):
    """
    Estima a matriz de homografia ou uma transformação usando as correspondências encontradas.

    :param keypoints1: Keypoints da primeira imagem.
    :param keypoints2: Keypoints da segunda imagem.
    :param matches: Correspondências entre keypoints.
    :return: Matriz de transformação (homografia ou essencial).
    """
    # Extrair os pontos chave correspondentes com base nos índices
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Calcular a matriz essencial
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # Recuperar a rotação e translação a partir da matriz essencial
    _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts)
    
    return R, t


def draw_matches(img1, keypoints1, img2, keypoints2, matches, mask=None):
    """
    Desenha as correspondências entre duas imagens.

    :param img1: A primeira imagem.
    :param keypoints1: Keypoints da primeira imagem.
    :param img2: A segunda imagem.
    :param keypoints2: Keypoints da segunda imagem.
    :param matches: Correspondências entre keypoints.
    :param mask: Máscara opcional para filtrar correspondências.
    :return: Imagem com as correspondências desenhadas.
    """
    # Desenha correspondências
    if mask is None:
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)
    else:
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=mask.ravel().tolist(), flags=2)

    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, **draw_params)

    return img_matches
