# # 다양한 종류의 물체에서 특정 물체를 검출하여 사각형으로 표시하기
#
# import cv2
# import numpy as np
#
# img1 = cv2.imread('books/all_a.jpg')
# gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2 = cv2.imread('books/all.jpg')
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#
# sift = cv2.SIFT_create()
# kp1, des1 = sift.detectAndCompute(gray1, None)
# kp2, des2 = sift.detectAndCompute(gray2, None)
#
# flann_matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
# knn_match = flann_matcher.knnMatch(des1, des2, 2)  # 최근접 2개
#
# T = 0.7
# good_match = []
# for nearest1, nearest2 in knn_match:
#     if (nearest1.distance / nearest2.distance) < T:
#         good_match.append(nearest1)
#
# # good_match를 찾음
# # ==========================================================
# # good_match 특징점의 위치
# # queryIdx : 이미지1에서의 특징점 번호 / trainIdx : 이미지2에서의 특징 위치
# points1 = np.float32([kp1[gm.queryIdx].pt for gm in good_match])
# points2 = np.float32([kp2[gm.trainIdx].pt for gm in good_match])
#
# # homography를 계산
# H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
#
# h1, w1 = img1.shape[0], img1.shape[1]  # 첫 번째 영상의 크기(검색 이미지)
# h2, w2 = img2.shape[0], img2.shape[1]  # 두 번째 영상의 크기
#
# # homography가 적용된 위치 계산
# # 주어진 이미지1에 대해 변환시킬려고 함
# box1 = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]).reshape(4, 1, 2)
# box2 = cv2.perspectiveTransform(box1, H)  # 사각형은 아니고 4개의 꼭짓점을 가진 다각형이므로 -> polyline으로 그림
#
# # 다각형으로 그림
# img2 = cv2.polylines(img2, [np.int32(box2)], True, (0, 255, 0), 8)
#
# img_match = np.empty((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
# cv2.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#
#
#
# cv2.imshow('Matches and Homography', img_match)
#
# cv2.waitKey()
# cv2.destroyAllWindows()

import cv2
import numpy as np

# 이미지 읽기
img1 = cv2.imread('books/all_d.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('books/all.jpg')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# SIFT 특징점 검출 및 기술자 생성
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# FLANN 기반 매칭
flann_matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_match = flann_matcher.knnMatch(des1, des2, 2)  # 최근접 2개

# 좋은 매칭 추출
T = 0.7
good_match = []
for nearest1, nearest2 in knn_match:
    if (nearest1.distance / nearest2.distance) < T:
        good_match.append(nearest1)

# 매칭된 특징점들의 위치 추출
points1 = np.float32([kp1[gm.queryIdx].pt for gm in good_match])
points2 = np.float32([kp2[gm.trainIdx].pt for gm in good_match])

# Homography 계산
H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

# 첫 번째 이미지의 크기
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# 첫 번째 이미지의 네 모서리 좌표를 변환
box1 = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]).reshape(4, 1, 2)
box2 = cv2.perspectiveTransform(box1, H)

# 변환된 영역을 다각형으로 그리기
img2 = cv2.polylines(img2, [np.int32(box2)], True, (0, 255, 0), 8)

# 매칭된 결과 이미지를 생성
img_match = np.empty((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
cv2.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 이미지 크기 조정 (화면에 출력하기 전 축소)
resize_factor = 0.2  # 50% 크기로 축소
img_match_resized = cv2.resize(img_match, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

# 이미지 출력
cv2.imshow('Matches and Homography_d', img_match_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
