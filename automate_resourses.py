###     www.youtube.com/c/programminghutofficial

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageGrab
import pyautogui
import time


start = time.time()
while True:
    end = time.time()
    if int(end - start) == 120:
        print("inif")
        start = end
        test = np.array(ImageGrab.grab(bbox=(16,62,1028,630)))
        test = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
        
        train_elix = cv2.imread('collect/elix.jpg', 0)
        train_gold = cv2.imread('collect/gold.jpg', 0)
        
        #############   elix collection ###################
        sift = cv2.xfeatures2d.SIFT_create()        
        kp_elix1, des_elix1 = sift.detectAndCompute(train_elix,None)
        kp_elix2, des_elix2 = sift.detectAndCompute(test,None)        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_elix1,des_elix2, k=2)
        
        match_pts_elix = []
        for m1, m2 in matches:
            if m1.distance < 0.65*m2.distance:
                idx = m1.trainIdx
                match_pts_elix.append(kp_elix2[idx].pt)
          
        if len(match_pts_elix) != 0:
            match_pts_elix = np.array(match_pts_elix)
            pyautogui.click(match_pts_elix[0, 0]+16, match_pts_elix[0, 1]+62, button='left')
        
        else:
            print("sorry ! no elixers farmed yet")
        
      ###   (x, y, 'left')  
        ################  gold collection #############################
        sift2 = cv2.xfeatures2d.SIFT_create()        
        kp_gold1, des_gold1 = sift2.detectAndCompute(train_gold,None)
        kp_gold2, des_gold2 = sift2.detectAndCompute(test,None)        
        bf2 = cv2.BFMatcher()
        matches2 = bf.knnMatch(des_gold1,des_gold2, k=2)
        
        match_pts_gold = []
        for m1, m2 in matches2:
            if m1.distance < 0.70*m2.distance:
                idx = m1.trainIdx
                match_pts_gold.append(kp_gold2[idx].pt)
          
        if len(match_pts_gold) != 0:
            match_pts_gold = np.array(match_pts_gold)
            pyautogui.click(match_pts_gold[0, 0]+16, match_pts_gold[0, 1]+62, button='left')
        
        else:
            print("sorry ! no gold farmed yet")
        
        if cv2.waitKey(1) == 27:
            break

        
        
        
        
        
        
        
        
        
        
        
        


