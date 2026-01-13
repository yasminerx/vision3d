import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from pathlib import Path
from scipy.optimize import least_squares

def detect_and_match_features(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize the feature detector and extractor (e.g., SIFT)
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Initialize the feature matcher using brute-force matching
    bf = cv2.BFMatcher()

    # Match the descriptors using brute-force matching
    matches = bf.match(descriptors1, descriptors2)

    # Select the top N matches
    num_matches = 30
    matches = sorted(matches, key=lambda x: x.distance)[:num_matches]

    return keypoints1, keypoints2, matches

def estimate_homography(keypoints1, keypoints2, matches, threshold=3):
    # Convert keypoints from matches into source and destination points arrays
    # For each match, queryIdx corresponds to keypoints in the first image
    # and trainIdx corresponds to keypoints in the second image
    src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute the homography matrix using RANSAC algorithm to handle outliers
    # This matrix transforms points from the first image plane to the second image plane
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, threshold)
    
    # H is the homography matrix, mask indicates inliers and outliers
    return H, mask

# H, mask = estimate_homography(keypoints1, keypoints2, matches)

def warp_images(img1, img2, H):
    # Get dimensions of both images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Define corners of both images
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # Use homography to transform corners of the second image
    warped_corners2 = cv2.perspectiveTransform(corners2, H)

    # Combine corners to find size of resulting panorama
    all_corners = np.concatenate((corners1, warped_corners2), axis=0)

    # Find bounding box of all corners
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Adjust translation to ensure the image fits within the calculated size
    translation_dist = [-xmin, -ymin]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]], dtype=np.float32)

    # Warp images to the panorama's coordinate system
    result_size = (xmax - xmin, ymax - ymin)
    warped_img1 = cv2.warpPerspective(img1, H_translation.dot(H), result_size)
    warped_img2 = cv2.warpPerspective(img2, H_translation, result_size)

    mask1 = np.where(warped_img1 > 0, 255, 0).astype(np.uint8)
    
    panorama = np.where(mask1, warped_img1, warped_img2)

    return panorama, warped_img1, warped_img2
    
# warped_img,_,_ = warp_images(img1, img2, H)

def glue_in_half(la,lb,rows,cols,dpt,img1):
        return np.hstack((la[:,0:cols//2], lb[:,cols//2:]))

def multi_band_blending(img1, img2, levels, glue_function):
    # generate Gaussian pyramid for A
    G = img1.copy()
    gpA = [G]
    for i in range(levels):  # Utiliser `levels` au lieu de 6
        G = cv2.pyrDown(G)
        gpA.append(G)

    # generate Gaussian pyramid for B
    G = img2.copy()
    gpB = [G]
    for i in range(levels):  # Utiliser `levels` au lieu de 6
        G = cv2.pyrDown(G)
        gpB.append(G)

    # generate Laplacian Pyramid for A
    lpA = [gpA[levels-1]]  # Adapter l'index au nombre de `levels`
    for i in range(levels-1, 0, -1):  # Adapter la boucle au nombre de `levels`
        GE = cv2.pyrUp(gpA[i])
        GE = cv2.resize(GE, (gpA[i-1].shape[1], gpA[i-1].shape[0]))
        L = cv2.subtract(gpA[i-1], GE)
        lpA.append(L)

    # generate Laplacian Pyramid for B
    lpB = [gpB[levels-1]]  # Adapter l'index au nombre de `levels`
    for i in range(levels-1, 0, -1):  # Adapter la boucle au nombre de `levels`
        GE = cv2.pyrUp(gpB[i])
        GE = cv2.resize(GE, (gpB[i-1].shape[1], gpB[i-1].shape[0]))
        L = cv2.subtract(gpB[i-1], GE)
        lpB.append(L)

    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = glue_function(la,lb,rows,cols,dpt,img1)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, levels):  # Adapter la boucle au nombre de `levels`
        ls_ = cv2.pyrUp(ls_)
        # Assurer que la taille de `ls_` correspond à celle de `LS[i]` avant l'addition
        ls_ = cv2.resize(ls_, (LS[i].shape[1], LS[i].shape[0]))
        ls_ = cv2.add(ls_, LS[i])
    return ls_

# A = cv2.imread('test/l.jpg')
# B = cv2.imread('test/r.jpg')

# AB_blend = multi_band_blending(A, B, 6 , glue_in_half)
# plt.imshow(cv2.cvtColor(AB_blend, cv2.COLOR_BGR2RGB))
# plt.show()

def calc_middle(img):
        # Trouver les indices de début et de fin de la zone non nulle dans img redimensionnée
        non_zero_columns = np.any(img > 0, axis=(0, 2))
        start, end = np.where(non_zero_columns)[0][[0, -1]]

        # Calculer le milieu de la zone de chevauchement
        overlap_middle = (start + end) // 2

        return start, end, overlap_middle

def glue_in_mid_overlap(la,lb,rows,cols,dpt,img1):
        
    resized_img1 = cv2.resize(img1, (cols, rows))
    start, end, overlap_middle = calc_middle(resized_img1)

    la_left_blend = la[:, start:overlap_middle]
    lb_right_blend = lb[:, overlap_middle:end]
    
    # Combiner les parties fusionnées
    combined = np.zeros((rows, cols, dpt), dtype='float32')
    combined[:, start:overlap_middle] = la_left_blend
    combined[:, overlap_middle:end] = lb_right_blend
    
    # Conversion des résultats en CV_8U après l'alpha-blending
    combined = np.clip(combined, 0, 255).astype('uint8')

    return combined

def alpha_none(bend_overlap,overlap_img1,overlap_img2): 
    return bend_overlap

def blend_images(img1, img2, H, alpha_func, pyramid_levels=2):
    warped_img, warped_img1, warped_img2 = warp_images(img1,img2,H)

    mask1 = np.where(warped_img1 > 0, 255, 0).astype(np.uint8)
    mask2 = np.where(warped_img2 > 0, 255, 0).astype(np.uint8)
    overlap_mask = np.bitwise_and(mask1, mask2)
  
    overlap_img1 = cv2.bitwise_and(warped_img1, overlap_mask)
    overlap_img2 = cv2.bitwise_and(warped_img2, overlap_mask)

    bend_overlap = multi_band_blending(overlap_img1, overlap_img2, pyramid_levels, glue_in_mid_overlap)
    bend_overlap = alpha_func(bend_overlap,overlap_img1,overlap_img2)

    bend_img = warped_img
    bend_img = np.where(overlap_mask == 255, bend_overlap, bend_img)
    
    return bend_img
    
#warped_img = blend_images(img1, img2, H, alpha_none)

## Show the image with matplotlib
# plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
# # plt.imshow(warped_img)
# plt.show()


def alpha_blending(bend_overlap,overlap_img1,overlap_img2):
        rows, cols, dpt = overlap_img1.shape
        start, end, overlap_middle = calc_middle(overlap_img1)

        # Ajustement des gradients d'opacité pour qu'ils soient appliqués uniquement sur la zone non nulle
        alpha_left = np.zeros((rows, overlap_middle - start, 1))
        alpha_right = np.zeros((rows, end - overlap_middle, 1))
        
        if overlap_middle - start > 0:
            gradient_left = np.linspace(0, 1, overlap_middle - start)[np.newaxis, :, np.newaxis]**(5)
            alpha_left = np.tile(gradient_left, (rows, 1, 1))
            
        if end - overlap_middle > 0:
            gradient_right = np.linspace(1, 0, end - overlap_middle)[np.newaxis, :, np.newaxis]**(5)
            alpha_right = np.tile(gradient_right, (rows, 1, 1))
            
        # Appliquer les gradients d'opacité pour fusionner les images
        la_left_blend = overlap_img1[:, start:overlap_middle] * (1-alpha_left) + bend_overlap[:, start:overlap_middle] * (alpha_left)
        lb_right_blend = bend_overlap[:, overlap_middle:end] * (alpha_right) + overlap_img2[:, overlap_middle:end] * (1-alpha_right)

        # Combiner les parties fusionnées
        combined = np.zeros((rows, cols, dpt), dtype='float32')
        combined[:, start:overlap_middle] = la_left_blend
        combined[:, overlap_middle:end] = lb_right_blend
        
        # Conversion des résultats en CV_8U après l'alpha-blending
        combined = np.clip(combined, 0, 255).astype('uint8')

        return combined



def check_interior_exterior(mask, cropping_rect):
    x, y, w, h = cropping_rect
    
    # Vérifier les bornes
    if w <= 0 or h <= 0:
        return True, (0, 0, 0, 0)
    
    sub = mask[y:y+h, x:x+w]
    top = bottom = left = right = 0

    # Check top and bottom
    if np.any(sub[0, :] == 0):
        top = 1
    if np.any(sub[-1, :] == 0):
        bottom = 1

    # Check left and right
    if np.any(sub[:, 0] == 0):
        left = 1
    if np.any(sub[:, -1] == 0):
        right = 1

    finished = (top, bottom, left, right) == (0, 0, 0, 0)
    return finished, (top, bottom, left, right)

def crop_max_content_area(source):
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return source  # Retourne l'image originale si aucun contour n'est trouvé

    # Find largest contour
    contour = max(contours, key=cv2.contourArea)
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

    xs = [point[0][0] for point in contour]
    ys = [point[0][1] for point in contour]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    cropping_rect = (x_min, y_min, x_max - x_min, y_max - y_min)
    finished = False

    while not finished:
        finished, (top, bottom, left, right) = check_interior_exterior(contour_mask, cropping_rect)

        if top:
            y_min += 1
        if bottom:
            y_max -= 1
        if left:
            x_min += 1
        if right:
            x_max -= 1

        cropping_rect = (x_min, y_min, x_max - x_min, y_max - y_min)

    cropped_image = source[y_min:y_max, x_min:x_max]

    return cropped_image

# Test images (5 images for testing only - real datasets may have 30-40+ images)
test_img1 = cv2.imread('test/im01.jpg')
test_img2 = cv2.imread('test/im02.jpg')
test_img3 = cv2.imread('test/im03.jpg')
test_img4 = cv2.imread('test/im04.jpg')
test_img5 = cv2.imread('test/im05.jpg')

def two_img_stitching(img1,img2,crop=False,pyramid_levels=2):
    keypoints1, keypoints2, matches = detect_and_match_features(img1, img2)
    H, mask = estimate_homography(keypoints1, keypoints2, matches)
    panorama = blend_images(img1, img2, H, alpha_blending, pyramid_levels)
    if crop : panorama = crop_max_content_area(panorama)
    return panorama

def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def resize_max_width(img, max_width=600):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_h = int(h * scale)
        return cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)
    return img

def stitch_hierarchical(images, crop_final=True, max_width=300, group_size=2):
    """
    Assemble les images de manière hiérarchique (divide and conquer).
    
    Au lieu de faire img1+img2=img12, img12+img3=img123, etc.
    On fait des groupes: img12, img34, img56, etc.
    Puis on fusionne les groupes: img1234, img5678, etc.
    
    Args:
        images: Liste d'images à assembler
        crop_final: Couper le panorama final
        max_width: Largeur maximale pour le redimensionnement
        group_size: Taille des groupes à assembler ensemble
    """
    if len(images) == 0:
        raise ValueError("Need at least 1 image")
    
    if len(images) == 1:
        return resize_max_width(images[0], max_width)
    
    # Redimensionner toutes les images d'abord
    images = [resize_max_width(img, max_width) for img in images]
    print(f"[INFO] Assemblage hiérarchique de {len(images)} images (groupes de {group_size})")
    
    current_level = images.copy()
    level_num = 1
    
    while len(current_level) > 1:
        print(f"[INFO] Niveau {level_num}: {len(current_level)} images à fusionner")
        next_level = []
        
        # Diviser en groupes
        for i in range(0, len(current_level), group_size):
            group = current_level[i:i+group_size]
            
            if len(group) == 1:
                next_level.append(group[0])
            else:
                # Assembler séquentiellement dans ce groupe
                print(f"[INFO]   Groupe {i//group_size + 1}: assemblage de {len(group)} images")
                pano = group[0]
                for j, img in enumerate(group[1:], start=2):
                    pano = two_img_stitching(pano, img, crop=False, pyramid_levels=2)
                    print(f"[INFO]     Image {j}/{len(group)} ajoutée, taille avant resize: {pano.shape}")
                    pano = resize_max_width(pano, max_width)
                    print(f"[INFO]     Après resize: {pano.shape}")
                
                # CRITIQUE: Redimensionner le panorama de groupe avant de l'ajouter au niveau suivant
                pano = resize_max_width(pano, max_width)
                next_level.append(pano)
        
        # CRITIQUE: Redimensionner tous les panoramas du niveau suivant
        current_level = [resize_max_width(p, max_width) for p in next_level]
        level_num += 1
    
    # Crop final uniquement à la toute fin
    result = current_level[0]
    if crop_final:
        print("[INFO] Crop final du panorama")
        result = crop_max_content_area(result)
    
    print(f"[INFO] Panorama final: {result.shape}")
    return result

def detect_all_features_and_matches(images, max_distance=5):
    """
    Détecte les features dans toutes les images et crée le graphe de correspondances.
    
    Args:
        images: Liste des images
        max_distance: Distance maximale entre images pour chercher des matches (pour loop closure)
    
    Returns:
        - all_keypoints: Liste des keypoints pour chaque image
        - all_descriptors: Liste des descriptors pour chaque image
        - matches_graph: Dict {(i,j): matches} pour toutes les paires d'images avec overlap
    """
    all_keypoints = []
    all_descriptors = []
    sift = cv2.SIFT_create()
    
    print("[INFO] Détection des features dans toutes les images...")
    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        all_keypoints.append(kp)
        all_descriptors.append(desc)
        print(f"  Image {i}: {len(kp)} keypoints détectés")
    
    # Matcher les images avec overlap potentiel (consécutives + loop closures)
    matches_graph = {}
    bf = cv2.BFMatcher()
    
    print(f"[INFO] Matching entre images (distance max: {max_distance})...")
    for i in range(len(images)):
        for j in range(i+1, min(i+max_distance+1, len(images))):
            if all_descriptors[i] is None or all_descriptors[j] is None:
                continue
            
            # Utiliser knnMatch avec ratio test de Lowe
            knn_matches = bf.knnMatch(all_descriptors[i], all_descriptors[j], k=2)
            
            # Ratio test de Lowe pour filtrer les bons matches
            good_matches = []
            for match_pair in knn_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:  # Ratio test
                        good_matches.append(m)
            
            if len(good_matches) >= 10:  # Minimum 10 bons matches
                # Vérifier avec RANSAC que l'homographie est valide
                src_pts = np.float32([all_keypoints[i][m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
                dst_pts = np.float32([all_keypoints[j][m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
                
                H_test, mask_test = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H_test is not None:
                    # Compter les inliers
                    inliers_count = np.sum(mask_test)
                    if inliers_count >= 8:  # Au moins 8 inliers
                        # Ne garder que les inliers
                        inlier_matches = [good_matches[idx] for idx in range(len(good_matches)) if mask_test[idx]]
                        matches_graph[(i, j)] = inlier_matches
                        print(f"  Images {i}-{j}: {len(inlier_matches)} matches (inliers/{len(good_matches)} total)")
    
    return all_keypoints, all_descriptors, matches_graph

def homography_to_params(H):
    """Convertit une matrice 3x3 en vecteur de 8 paramètres (on fixe h33=1)"""
    return np.array([H[0,0], H[0,1], H[0,2], H[1,0], H[1,1], H[1,2], H[2,0], H[2,1]])

def params_to_homography(params):
    """Reconstruit une matrice 3x3 depuis 8 paramètres"""
    H = np.array([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
        [params[6], params[7], 1.0]
    ])
    return H

def bundle_adjustment_residuals(params, all_keypoints, matches_graph, n_images):
    """
    Calcule les résidus pour le bundle adjustment.
    
    Graphe:
        - Nœuds: Tous les coefficients des homographies H_i (8 params chacune)
        - Arêtes: Contraintes par les équations de transformation p'_j = H_i * p_i
        
    Args:
        params: Vecteur de tous les paramètres des homographies
        all_keypoints: Liste des keypoints pour chaque image
        matches_graph: Dict {(i,j): matches}
        n_images: Nombre d'images
    
    Returns:
        residuals: Vecteur des erreurs de reprojection
    """
    # Reconstruire les homographies depuis params
    homographies_dict = {}
    param_idx = 0
    
    for (img_i, img_j) in sorted(matches_graph.keys()):
        H_params = params[param_idx:param_idx+8]
        H = params_to_homography(H_params)
        homographies_dict[(img_i, img_j)] = H
        param_idx += 8
    
    residuals = []
    
    # Pour chaque paire d'images avec des matches
    for (img_i, img_j), matches in matches_graph.items():
        H = homographies_dict[(img_i, img_j)]
        
        for match in matches:
            # Point dans image i
            pt_i = np.array([all_keypoints[img_i][match.queryIdx].pt[0],
                            all_keypoints[img_i][match.queryIdx].pt[1],
                            1.0])
            
            # Point correspondant dans image j
            pt_j_actual = np.array([all_keypoints[img_j][match.trainIdx].pt[0],
                                   all_keypoints[img_j][match.trainIdx].pt[1]])
            
            # Projection de pt_i dans image j via H
            pt_j_proj = H @ pt_i
            
            # Vérifier la validité de la projection
            if abs(pt_j_proj[2]) < 1e-8:
                # Point à l'infini, pénaliser fortement
                residuals.extend([100.0, 100.0])
            else:
                pt_j_proj = pt_j_proj[:2] / pt_j_proj[2]  # Normalisation homogène
                
                # Erreur de reprojection
                error = pt_j_actual - pt_j_proj
                residuals.extend([error[0], error[1]])
    
    return np.array(residuals)

def stitch_with_bundle_adjustment(images, crop_final=True, max_width=1000, max_match_distance=1):
    """
    Assemble un panorama en utilisant Bundle Adjustment - VERSION CORRECTE.
    Inspiré de https://github.com/aartighatkesar/Image-Mosaicing
    
    Approche:
    1. Calculer les homographies entre paires consécutives (avec optimisation LM)
    2. Composer toutes les homographies vers l'IMAGE CENTRALE (pas la première)
    3. Assembler sur un canvas global
    
    Cela évite l'accumulation d'erreurs de la méthode séquentielle.
    """
    # Redimensionner les images
    images = [resize_max_width(img, max_width) for img in images]
    n_images = len(images)
    
    if n_images == 0:
        return None
    if n_images == 1:
        return images[0]
    
    # Image de référence = celle du milieu
    middle_idx = n_images // 2
    print(f"[INFO] Utilisation de l'image {middle_idx} comme référence centrale")
    
    # Détecter features et matches
    all_keypoints, all_descriptors, matches_graph = detect_all_features_and_matches(
        images, max_distance=max_match_distance
    )
    
    if len(matches_graph) == 0:
        print("[ERREUR] Aucun match trouvé!")
        return images[0]
    
    # Étape 1: Calculer les homographies entre images consécutives
    print("[INFO] Calcul des homographies entre paires consécutives...")
    H_pairwise = {}  # H_{i,i+1} transforme image_i vers image_i+1
    
    for i in range(n_images - 1):
        if (i, i+1) in matches_graph:
            kp1 = all_keypoints[i]
            kp2 = all_keypoints[i+1]
            matches = matches_graph[(i, i+1)]
            
            # Estimation initiale par RANSACmax_ma
            H_init, _ = estimate_homography(kp1, kp2, matches, threshold=3)
            if H_init is None:
                print(f"  [WARNING] H_{i}→{i+1}: échec, utilisation identité")
                H_init = np.eye(3)
            
            # Optimisation de l'homographie par Levenberg-Marquardt
            # Utiliser tous les inliers pour raffiner
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
            correspondence = np.hstack([src_pts, dst_pts])  # [x1, y1, x2, y2]
            
            # Optimiser avec scipy (plus stable que notre implémentation)
            from scipy.optimize import least_squares
            
            def residuals_H(h_flat, corr):
                H = h_flat.reshape(3, 3)
                src = np.column_stack([corr[:, :2], np.ones(len(corr))])
                dst_pred = (H @ src.T).T
                dst_pred = dst_pred[:, :2] / dst_pred[:, 2:3]
                dst_actual = corr[:, 2:4]
                return (dst_actual - dst_pred).ravel()
            
            result = least_squares(
                residuals_H,
                H_init.ravel(),
                args=(correspondence,),
                method='lm',
                max_nfev=50
            )
            
            H_optimized = result.x.reshape(3, 3)
            H_optimized = H_optimized / H_optimized[2, 2]  # Normaliser
            H_pairwise[(i, i+1)] = H_optimized
            print(f"  H_{i}→{i+1} calculée et optimisée")
        else:
            print(f"  [WARNING] Pas de match entre {i} et {i+1}, utilisation identité")
            H_pairwise[(i, i+1)] = np.eye(3)
    
    # Étape 2: Composer les homographies vers l'image centrale
    print(f"[INFO] Composition des homographies vers image centrale ({middle_idx})...")
    H_to_middle = {}
    H_to_middle[middle_idx] = np.eye(3)
    
    # Images à gauche de la centrale: composer vers la droite
    # H_{i→middle} = H_{i→i+1} * H_{i+1→i+2} * ... * H_{middle-1→middle}
    for i in range(middle_idx):
        H_accum = np.eye(3)
        for j in range(i, middle_idx):
            if (j, j+1) in H_pairwise:
                H_accum = H_pairwise[(j, j+1)] @ H_accum
        H_to_middle[i] = H_accum
        print(f"  H_{i}→{middle_idx} composée")
    
    # Images à droite de la centrale: composer vers la gauche (inverse)
    # H_{i→middle} = inv(H_{middle→middle+1}) * inv(H_{middle+1→middle+2}) * ... * inv(H_{i-1→i})
    for i in range(middle_idx + 1, n_images):
        H_accum = np.eye(3)
        for j in range(i - 1, middle_idx - 1, -1):
            if (j, j+1) in H_pairwise:
                H_accum = np.linalg.inv(H_pairwise[(j, j+1)]) @ H_accum
        H_to_middle[i] = H_accum
        print(f"  H_{i}→{middle_idx} composée")
    
    # Étape 3: Calculer la taille du canvas nécessaire
    print("[INFO] Calcul de la taille du canvas...")
    h_ref, w_ref = images[middle_idx].shape[:2]
    
    min_x, min_y = 0, 0
    max_x, max_y = w_ref, h_ref
    
    for i in range(n_images):
        if i == middle_idx:
            continue
        
        h, w = images[i].shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, H_to_middle[i])
        
        min_x = min(min_x, transformed[:, 0, 0].min())
        min_y = min(min_y, transformed[:, 0, 1].min())
        max_x = max(max_x, transformed[:, 0, 0].max())
        max_y = max(max_y, transformed[:, 0, 1].max())
    
    # Offset pour translation
    offset_x = int(np.floor(-min_x))
    offset_y = int(np.floor(-min_y))
    canvas_width = int(np.ceil(max_x - min_x))
    canvas_height = int(np.ceil(max_y - min_y))
    
    print(f"  Taille canvas: {canvas_width} x {canvas_height}")
    print(f"  Offset: ({offset_x}, {offset_y})")
    
    # Vérification sanité
    if canvas_width > 15000 or canvas_height > 15000:
        print(f"[ERREUR] Canvas trop grand: {canvas_width}x{canvas_height}")
        print("  Fallback vers méthode hiérarchique...")
        return stitch_hierarchical(images, crop_final=crop_final, max_width=max_width, group_size=2)
    
    # Matrice de translation
    H_translation = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)
    
    # Étape 4: Assembler progressivement sur le canvas avec blending
    print("[INFO] Assemblage sur canvas avec alpha blending...")
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    weight_canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    
    # Assembler dans l'ordre: centrale d'abord, puis alternativement gauche/droite
    assembly_order = [middle_idx]
    left = middle_idx - 1
    right = middle_idx + 1
    
    while left >= 0 or right < n_images:
        if left >= 0:
            assembly_order.append(left)
            left -= 1
        if right < n_images:
            assembly_order.append(right)
            right += 1
    
    for idx, i in enumerate(assembly_order):
        H_final = H_translation @ H_to_middle[i]
        warped = cv2.warpPerspective(images[i], H_final, (canvas_width, canvas_height))
        
        # Créer un masque avec feathering pour cette image
        h, w = images[i].shape[:2]
        weight_img = np.ones((h, w), dtype=np.float32)
        
        # Feathering aux bords (transitions douces)
        feather_size = min(50, w//10, h//10)
        for j in range(feather_size):
            alpha = j / feather_size
            weight_img[j, :] *= alpha
            weight_img[-j-1, :] *= alpha
            weight_img[:, j] *= alpha
            weight_img[:, -j-1] *= alpha
        
        # Transformer le masque de poids
        warped_weight = cv2.warpPerspective(weight_img, H_final, (canvas_width, canvas_height))
        
        # Accumuler avec pondération
        for c in range(3):
            canvas[:, :, c] += warped[:, :, c].astype(np.float32) * warped_weight
        weight_canvas += warped_weight
        
        print(f"  Image {i} ajoutée au canvas (ordre: {idx+1}/{len(assembly_order)})")
    
    # Normaliser par les poids pour obtenir la moyenne pondérée
    mask_valid = weight_canvas > 1e-6
    for c in range(3):
        canvas[:, :, c][mask_valid] /= weight_canvas[mask_valid]
    
    # Convertir en uint8
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    
    # IMPORTANT: Ne pas crop pour garder les pixels noirs et montrer la vraie forme
    if crop_final:
        print("[INFO] Crop léger du panorama (conserve les pixels noirs aux bords)...")
        # Crop seulement les lignes/colonnes complètement noires
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        rows_with_content = np.any(gray > 0, axis=1)
        cols_with_content = np.any(gray > 0, axis=0)
        
        if np.any(rows_with_content) and np.any(cols_with_content):
            first_row = np.argmax(rows_with_content)
            last_row = len(rows_with_content) - np.argmax(rows_with_content[::-1])
            first_col = np.argmax(cols_with_content)
            last_col = len(cols_with_content) - np.argmax(cols_with_content[::-1])
            
            canvas = canvas[first_row:last_row, first_col:last_col]
    
    print(f"[INFO] Panorama final: {canvas.shape}")
    return canvas


images= load_images_from_folder('vision3d/images_test/city/')

print(f"\n{'='*70}")
print("Bundle Adjustment pour panorama multi-images")
print(f"Chargé {len(images)} frames")
print(f"{'='*70}\n")

final_panorama = stitch_with_bundle_adjustment(
    images, 
    crop_final=True, 
    max_width=1000,
    max_match_distance=3
)

if final_panorama is not None and final_panorama.size > 0:
    plt.imshow(cv2.cvtColor(final_panorama, cv2.COLOR_BGR2RGB))
plt.title(f"Panorama from {len(images)} images")
plt.show()