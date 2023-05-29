from tkinter import *
from tkinter import filedialog
import numpy as np
import cv2
from copy import deepcopy

MIN_MATCH_COUNT = 4

def imgSticher_4(imgs):

    resultImg = imgs[0]
    for i in range(1, len(imgs)):
        cv2.imshow(f"qImg", resultImg)
        cv2.imshow(f"tImg", imgs[i])
        cv2.waitKey(0)
        qImgb, qImgg, qImgr = cv2.split(resultImg)
        tImgb, tImgg, tImgr = cv2.split(imgs[i])

        matchImgb, resultImgb = imgRegistration(qImgb, tImgb)
        matchImgg, resultImgg = imgRegistration(qImgg, tImgg)
        matchImgr, resultImgr = imgRegistration(qImgr, tImgr)

        
        resultImg = cv2.merge([resultImgb, resultImgg, resultImgr])
        matchImg = cv2.merge([matchImgb, matchImgg, matchImgr])

        cv2.imshow(f"matchImgB{i}", matchImgb)
        cv2.imshow(f"matchImgG{i}", matchImgg)
        cv2.imshow(f"matchImgR{i}", matchImgr)
        cv2.imshow(f"resultImgB{i}", resultImgb)
        cv2.imshow(f"resultImgG{i}", resultImgg)
        cv2.imshow(f"resultImgR{i}", resultImgr)
        cv2.imshow(f"resultImg{i}", resultImg)

        cv2.waitKey(0)
    


    return resultImg

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
        if m.distance < 0.75*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) # -1 : auto size
        # print(src_pts.shape)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # print(dst_pts.shape)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        print(f"M :{M[0, 2], M[1, 2]}")
        
        matchesMask = mask.ravel().tolist()

        h, w = qImg.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

        theta = np.arctan(M[1, 2]/M[0, 2])

        # print(f"M : {M} \n Theta : {theta}")
                
        dst = cv2.perspectiveTransform(pts, M)
        '''
        tImg = cv2.polylines(tImg, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        '''

    else :
        print(f"Not enough matches are found - {len(good)} / {MIN_MATCH_COUNT}")
        matchesMask = None

    draw_params = dict(matchColor = (0, 255, 0),  # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
        

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params) # 키워드 가변인자
    

    # -- 배치
    dx, dy = M[0:2, 2]
    dx, dy = np.abs(np.ceil(dx)), np.abs(np.ceil(dy))
    dx, dy = tuple(map(int, [dx, dy]))

    if np.abs(theta)<0.17 : # 10도
        if M[0, 2] > 0: 
            dst = cv2.warpPerspective(qImg, M, (windowW, windowH))
            dst[0:tImg.shape[0], 0:tImg.shape[1]] = tImg
            print("Img1 : 오른쪽, Img2 : 왼쪽")
        else:
            
            M[0, 2] = M[0, 2] + dx
            dst = cv2.warpPerspective(qImg, M, (windowW, windowH))
            dst[0:tImg.shape[0], dx:dx+tImg.shape[1]] = tImg
            print("Img1 : 왼쪽, Img2 : 오른쪽")
    elif np.abs(theta)> 1.4: # 80도
        if M[1, 2] > 0:
            dst = cv2.warpPerspective(qImg, M, (windowW, windowH))
            dst[0:tImg.shape[0], 0:tImg.shape[1]] = tImg
            print("Img1 : 아래쪽, Img2 : 위쪽")
        else:
            M[1, 2] = M[1, 2] + dy
            dst = cv2.warpPerspective(qImg, M, (windowW, windowH))
            dst[dy:dy+tImg.shape[0], 0:tImg.shape[1]] = tImg
            print("Img1 : 위쪽, Img2 : 아래쪽")

    else: # 대각방향
        if M[1, 2] > 0 :
            if M[0, 2] >0:
                dst = cv2.warpPerspective(qImg, M, (windowW, windowH))
                dst[0:tImg.shape[0], 0:tImg.shape[1]] = tImg
                print("Img1 : 오른쪽 아래, Img2 : 왼쪽 위")
            else:
                M[0:2, 2] = M[0:2, 2] + np.array([dx, dy])
                dst = cv2.warpPerspective(qImg, M, (windowW, windowH))
                dst[dy:dy+tImg.shape[0], dx:dx+tImg.shape[1]] = tImg
                print("Img1 : 왼쪽 아래, Img2 : 오른쪽 위")
        else:
            if M[0, 2] >0:
                M[1, 2] = M[1, 2] + dy
                dst = cv2.warpPerspective(qImg, M, (windowW, windowH))
                dst[dy:dy+tImg.shape[0], 0:tImg.shape[1]] = tImg

                print("Img1 : 오른쪽 위, Img2 : 왼쪽 아래")
            else:
                M[0:2, 2] = M[0:2, 2] + np.array([dx, dy])
                dst = cv2.warpPerspective(qImg, M, (windowW, windowH))
                dst[dy:dy+tImg.shape[0], dx:dx+tImg.shape[1]] = tImg
                
                print("Img1 : 왼쪽 위, Img2 : 오른쪽 아래")


    cv2.destroyAllWindows()

    # 보정 수식
    dy = np.min(np.where(dst>0)[0])
    dx = np.min(np.where(dst>0)[1])
    M = np.float32([[1, 0, -dx],[0, 1, -dy]])
    
    dst = cv2.warpAffine(dst, M, (windowW, windowH))

    return img3, dst





if __name__ == "__main__":
    root = Tk()
    path = filedialog.askopenfilename(initialdir = "A.stiching", title= 'choose your image', filetypes = (("jpeg files", "*.jpg"), ("all files", "*.*")))
    img1 = cv2.imread(path)
    root.withdraw()

    imgW, imgH = img1.shape[1], img1.shape[0]
    windowW = 2*max(img1.shape[1], img1.shape[1])
    windowH = 2*max(img1.shape[0], img1.shape[0])

    root = Tk()
    path = filedialog.askopenfilename(initialdir = "A.stiching", title= 'choose your image', filetypes = (("jpeg files", "*.jpg"), ("all files", "*.*")))
    img2 = cv2.imread(path)
    root.withdraw()

    root = Tk()
    path = filedialog.askopenfilename(initialdir = "A.stiching", title= 'choose your image', filetypes = (("jpeg files", "*.jpg"), ("all files", "*.*")))
    img3 = cv2.imread(path)
    root.withdraw()

    root = Tk()
    path = filedialog.askopenfilename(initialdir = "A.stiching", title= 'choose your image', filetypes = (("jpeg files", "*.jpg"), ("all files", "*.*")))
    img4 = cv2.imread(path)
    root.withdraw()

    imgs = [img1, img2, img3, img4]
    resultImg = imgSticher_4(imgs)

    cv2.imshow("resultImg", resultImg)




    cv2.destroyAllWindows()
