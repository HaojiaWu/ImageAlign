from flask import Flask, render_template, request, send_from_directory, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads/')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/align', methods=['POST'])
def align_images():
    image1 = request.files.get('image1')
    image2 = request.files.get('image2')

    if not image1 or not image2:
        return jsonify(error="Both images are required for alignment"), 400

    image1_path = os.path.join(UPLOAD_FOLDER, image1.filename)
    image2_path = os.path.join(UPLOAD_FOLDER, image2.filename)

    try:
        image1.save(image1_path)
        image2.save(image2_path)
    except FileNotFoundError as e:
        return jsonify(error=f"Error saving file: {e}"), 500


    try:
        method = request.form.get('method', 'sift')

        if method == 'cross_correlation':
            aligned_image_path = align_using_cross_correlation(image1_path, image2_path)
        elif method == 'sift':
            aligned_image_path = align_using_sift(image1_path, image2_path)
        elif method == 'orb':
            aligned_image_path = align_using_orb(image1_path, image2_path)
        elif method == 'ecc':
            aligned_image_path = align_using_ecc(image1_path, image2_path)
        elif method == 'ransac':
            aligned_image_path = align_using_ransac(image1_path, image2_path)
        else:
            return jsonify(error="Invalid alignment method chosen"), 400
    except Exception as e:
        return jsonify(error=f"Error during alignment: {e}"), 500


    if method == 'sift':
        aligned_image_path = align_using_sift(image1_path, image2_path)
    elif method == 'orb':
        aligned_image_path = align_using_orb(image1_path, image2_path)
    elif method == 'ecc':
        aligned_image_path = align_using_ecc(image1_path, image2_path)
    elif method == 'ransac':
        aligned_image_path = align_using_ransac(image1_path, image2_path)
    else:
        return jsonify(error="Invalid alignment method chosen"), 400
    pass
    return send_from_directory(UPLOAD_FOLDER, os.path.basename(aligned_image_path))

def align_using_sift(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w, _ = img2.shape
    aligned = cv2.warpPerspective(img1, M, (w, h))

    aligned_image_path = os.path.join(UPLOAD_FOLDER, "aligned_sift.jpg")
    cv2.imwrite(aligned_image_path, aligned)

    return aligned_image_path

def align_using_orb(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    orb = cv2.ORB_create()
    
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    matches = sorted(matches, key=lambda x: x.distance)
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    aligned_img = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))
    result_path = os.path.join(UPLOAD_FOLDER, "aligned_orb.png")
    cv2.imwrite(result_path, aligned_img)
    
    return result_path

def align_using_ecc(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    warp_mode = cv2.MOTION_EUCLIDEAN
    
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    number_of_iterations = 5000
    
    termination_eps = 1e-10
    
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    
    _, warp_matrix = cv2.findTransformECC(img2, img1, warp_matrix, warp_mode, criteria)
    
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        aligned_img = cv2.warpPerspective(img1, warp_matrix, (img2.shape[1], img2.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        aligned_img = cv2.warpAffine(img1, warp_matrix, (img2.shape[1], img2.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    result_path = os.path.join(UPLOAD_FOLDER, "aligned_ecc.png")
    cv2.imwrite(result_path, aligned_img)
    
    return result_path

def align_using_ransac(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    orb = cv2.ORB_create()
    
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    
    aligned_img = cv2.warpAffine(img1, M, (img2.shape[1], img2.shape[0]))
    result_path = os.path.join(UPLOAD_FOLDER, "aligned_ransac.png")
    cv2.imwrite(result_path, aligned_img)
    
    return result_path


if __name__ == '__main__':
    app.run(debug=True)

