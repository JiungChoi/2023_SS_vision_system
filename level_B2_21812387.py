from tkinter import *
from tkinter import filedialog
import numpy as np
import cv2
from copy import deepcopy

MIN_MATCH_COUNT = 10

def generHomo(kp1, des1, kp2, des2, th):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) # (key = value)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe`s ratio test`
    good = []
    for m, n in matches :
        if m.distance < th*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) # -1 : auto size
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # print(f"M :{M}")
        
        
        cos = np.mean([M[0, 0], M[1, 1]])
        theta = ( np.arccos( cos ) ) * (180/np.pi) if np.abs(cos) < 1 else ( np.arccos( -cos ) ) * (180/np.pi)

        
        # if sin < 0:
        #     pass
        # elif sin >0:
        #     theta = 360 - theta
            
        
        # whatT = int(np.round(theta / 6)) if (M[0, 0] < 0) and (M[1, 1] <     0) else 60-int(np.round(theta / 6))
        
        # print(f"theta : \n{theta}")
        whatT = 60-int(np.round(theta / 6))
        
        print("Now Time is : ", whatT)
        matchesMask = mask.ravel().tolist()


    else :
        M, good, matchesMask, whatT = generHomo(kp1, des1, kp2, des2, th+0.1)


    return M, good, matchesMask, whatT

def whatTime(refImg, tarImg):
    qImg = deepcopy(refImg)
    tImg = deepcopy(tarImg)

    # 크기 정규화 과정
    ratioXT2Q = qImg.shape[1]/tImg.shape[1]
    ratioYT2Q = qImg.shape[0]/tImg.shape[0]

    tImg = cv2.resize(tImg, (0, 0), fx=ratioXT2Q, fy=ratioYT2Q)

    qImg = cv2.GaussianBlur(qImg, (5, 5), 20)
    tImg = cv2.GaussianBlur(tImg, (5, 5), 20)

    cv2.imshow("Img : Reference Img", qImg)
    cv2.imshow("Img : What time?", tImg)
    cv2.waitKey()

    

    # -- 특징, 디스크립터 추출
    # Initiate SIFT dector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(qImg, None)
    kp2, des2 = sift.detectAndCompute(tImg, None)

    
    M, good, matchesMask, whatT = generHomo(kp1, des1, kp2, des2, 0.4)
    draw_params = dict(matchColor = (0, 255, 0),  # draw matches in green color
                singlePointColor = None,
                matchesMask = matchesMask, # draw only inliers
                flags = 2)


    dst = cv2.warpPerspective(qImg, M, (windowW, windowH))
    matchedImg = cv2.drawMatches(qImg, kp1, tImg, kp2, good, None, **draw_params) # 키워드 가변인자

    # -- 분침 방향성분 추출 과정
    # 특징, 디스크립터 추출
    kp1, des1 = sift.detectAndCompute(qImg, None)
    kp2, des2 = sift.detectAndCompute(tImg, None)

    # cv2.imshow("result,", dst)

    return matchedImg, whatT





if __name__ == "__main__":
    root = Tk()
    path = filedialog.askopenfilename(initialdir = "B._WhatTime", title= 'choose your image', filetypes = (("all files", "*.*"), ("jpeg files", "*.jpg")))
    refImg = cv2.imread(path, 0)
    root.withdraw()

    root = Tk()
    path = filedialog.askopenfilename(initialdir = "B._WhatTime", title= 'choose your image', filetypes = (("all files", "*.*"), ("jpeg files", "*.jpg")))
    tarImg = cv2.imread(path, 0)
    root.withdraw()

    windowW = max(refImg.shape[1], tarImg.shape[1])
    windowH = max(refImg.shape[0], tarImg.shape[0])


    print("\n[2023 VisionSystem Termproject : Level_B2]", end="\n")
    matchImg, whatT = whatTime(refImg, tarImg)
    cv2.imshow("matched Img: Result", matchImg)
    print("")
    
    cv2.waitKey()
    cv2.destroyAllWindows()
