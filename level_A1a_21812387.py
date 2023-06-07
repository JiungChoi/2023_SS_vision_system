from tkinter import *
from tkinter import filedialog
import numpy as np
import cv2
from copy import deepcopy


MIN_MATCH_COUNT = 4


def imgRegistration(img1, img2):
    qImg = deepcopy(img1)
    tImg = deepcopy(img2)

    # -- 특징, 디스크립터 추출
    # Initiate SIFT dector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(qImg, None)
    kp2, des2 = sift.detectAndCompute(tImg, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) # (key = value)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe`s ratio test`
    good = []
    for m, n in matches :
        if m.distance < 0.9*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) # -1 : auto size
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        matchesMask = mask.ravel().tolist()

        theta = np.arctan(M[1, 2]/M[0, 2])

        # print(f"M : {M} \n Theta : {theta}")

        if np.abs(theta)<0.17 : # 10도
            if theta > 0: print("Img1 : 오른쪽, Img2 : 왼쪽")
            else: print("Img1 : 왼쪽, Img2 : 오른쪽")
        elif np.abs(theta)> 1.4: # 80도
            if theta > 0: print("Img1 : 아래쪽, Img2 : 위쪽")
            else: print("Img1 : 위쪽, Img2 : 아래쪽")

        else: # 대각방향
            if M[1, 2] > 0 :
                if M[0, 2] >0:
                    print("Img1 : 오른쪽 아래, Img2 : 왼쪽 위")
                else:
                    print("Img1 : 왼쪽 아래, Img2 : 오른쪽 위")
            else:
                if M[0, 2] >0:
                    print("Img1 : 오른쪽 위, Img2 : 왼쪽 아래")
                else:
                    print("Img1 : 왼쪽 위, Img2 : 오른쪽 아래")


        # h, w = qImg.shape
        # pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        # print(pts.shape)
        # dst = cv2.perspectiveTransform(pts, M)

        # tImg = cv2.polylines(tImg, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else :
        print(f"Not enough matches are found - {len(good)} / {MIN_MATCH_COUNT}")
        matchesMask = None

    draw_params = dict(matchColor = (0, 255, 0),  # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)


    matchedImg = cv2.drawMatches(qImg, kp1, tImg, kp2, good, None, **draw_params) # 키워드 가변인자
    

    cv2.destroyAllWindows()
    return matchedImg





if __name__ == "__main__":
    root = Tk()
    path = filedialog.askopenfilename(initialdir = "A._Stitching", title= 'choose your image', filetypes = (("jpeg files", "*.jpg"), ("all files", "*.*")))
    img1 = cv2.imread(path)
    root.withdraw()

    root = Tk()
    path = filedialog.askopenfilename(initialdir = "A._Stitching", title= 'choose your image', filetypes = (("jpeg files", "*.jpg"), ("all files", "*.*")))
    img2 = cv2.imread(path)
    root.withdraw()

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

   
    matchImg = imgRegistration(img1, img2)
    cv2.imshow("Img1", img1)
    cv2.imshow("Img2", img2)
    cv2.imshow("Matched Img", matchImg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()