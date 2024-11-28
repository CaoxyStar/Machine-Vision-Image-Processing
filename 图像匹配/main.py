"""
该程序为机器视觉课程图像匹配检测作业, 主要功能为基于SIFT、SURF、ORB方法进行特征点提取, 并计算单应性矩阵进行图像变换与拼接。
作者: 曹翔宇
日期: 2024年5月26日
操作系统: Windows 11
环境: Python 3.7
依赖包: opencv-python 3.4.2.16, numpy
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

# feature extract and match
def feature_matching(img1, img2, method='SIFT', use_ransac=False):
    if method == 'SIFT':
        # Initiate SIFT detector
        detector = cv2.xfeatures2d.SIFT_create()
    elif method == 'SURF':
        # Initiate SURF detector
        detector = cv2.xfeatures2d.SURF_create(400)
    elif method == 'ORB':
        # Initiate ORB detector
        detector = cv2.ORB_create()
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Find the keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # Match descriptors using FLANN matcher for SIFT and SURF, or BFMatcher for ORB
    if method in ['SIFT', 'SURF']:
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        good_matches = matcher.match(des1, des2)

    if use_ransac:
        # Extract location of good matches
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find the homography matrix using RANSAC
        M, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
    else:
        matches_mask = None
        M = None

    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, matchesMask=matches_mask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0, 255, 0))

    return img_matches, M

# stitch two image with homography matrix
def stitch_image(img1, img2, H):
    # height, width
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
	
	# get corners of two images
    img1_dims = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    img2_dims = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

	# transform one image with H
    img1_transform = cv2.perspectiveTransform(img1_dims, H)
	
	# get max and min corners
    result_dims = np.concatenate((img2_dims, img1_transform), axis=0)
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel()-0.5)
    [x_max, y_max ] = np.int32(result_dims.max(axis=0).ravel()+0.5)

    # set translation matrix
    transform_dist = [-x_min, -y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])
	
    # transform
    result_img = cv2.warpPerspective(img1, transform_array.dot(H), (x_max-x_min, y_max-y_min))

    # add the another image
    result_img[transform_dist[1]:transform_dist[1]+h2, 
                transform_dist[0]:transform_dist[0]+w2] = img2

    return result_img


def main():
    # Load images
    img1 = cv2.imread('src\A.png', cv2.IMREAD_COLOR)
    img2 = cv2.imread('src\B.png', cv2.IMREAD_COLOR)

    # Ensure images are loaded
    if img1 is None or img2 is None:
        raise FileNotFoundError("One of the input images is not found.")

    # Feature matching using SIFT, SURF, and ORB, both with and without RANSAC
    methods = ['SIFT', 'SURF', 'ORB']
    results = {}
    homographies = {}

    for method in methods:
        results[f'{method}_no_ransac'], _ = feature_matching(img1, img2, method=method, use_ransac=False)
        results[f'{method}_ransac'], homography = feature_matching(img1, img2, method=method, use_ransac=True)
        homographies[method] = homography
        print(f'Homography Matrix with {method}:')
        print(homography)

    # Save the feature match results
    for key, img in results.items():
        cv2.imwrite(f'dst/{key}.jpg', img)

    # Stitch image_1 and image_2
    stitched_images = {}
    for method in methods:
        if homographies[method] is not None:
            stitched_images[method] = stitch_image(img2, img1, homographies[method])

    # Save the stitched results
    for key, img in stitched_images.items():
        cv2.imwrite(f'dst/{key}_stitch.jpg', img)


if __name__ == "__main__":
    main()
