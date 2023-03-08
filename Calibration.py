"""Librerie utilizzate di supporto alle funzioni cui sotto :
- OpenCV
- Numpy
- Glob , che permette di gestire file presenti in una medesima
cartella che rispecchiano il medesimo pattern ( nel nome )
    """

import cv2
import numpy as np
import glob


"""Permette di normalizzare l'immagine , riducendo 
le differenze di luce presenti all'interno della scacchiera
e rimuovendo parzialmente il problema delle ombre """
def normalization(image):

    hh, ww = image.shape[:2]


    # illumination normalize
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # separate channels
    y, cr, cb = cv2.split(ycrcb)

    # get background which paper says (gaussian blur using standard deviation 5 pixel for 300x300 size image)
    # account for size of input vs 300
    sigma = int(5 * hh / 300)
    gaussian = cv2.GaussianBlur(y, (0, 0), sigma, sigma)

    # subtract background from Y channel
    y = (y - gaussian + 100)

    # merge channels back
    ycrcb = cv2.merge([y, cr, cb])

    # convert to BGR
    output = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return output


"""Permette di creare un set di immagini utili per la calibrazione
della camera , all'interno della quali dovrà comparire la scacchiera per
la calibrazione """
def save_img_for_calibration(webcam):

    i = 0

    while (True):

        success, img = webcam.read()

        if success:
            cv2.imshow("Calibrazione",img)

        if cv2.waitKey(1) & 0xFF == ord('t'):
            cv2.imwrite("./images/Photo" + str(i)+".jpg", img)
            i = i + 1

        elif cv2.waitKey(1) & 0xFF == ord('q'):
            # End the program
            break

"""Effettua la calibrazione della camera , restituendo una serie di 
parametri utili per rimuovere la distorsione dall'immagine"""
def camera_calibration():
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (6, 9)
    SIZE = 17
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, SIZE, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    images = glob.glob('./images/*.jpg')

    for fname in images:

        print(".",end=" ")
        cal_img = cv2.imread(fname)
        gray = cv2.cvtColor(cal_img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)
            cal_img = cv2.drawChessboardCorners(cal_img, CHECKERBOARD, corners2, ret)

        #cv2.imshow('img', cal_img)
        #cv2.waitKey(0)

    cv2.destroyAllWindows()

    h, w = cal_img.shape[:2]

    print("") # Giusto per stampare una nuova riga
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs , h , w


"""Rimuove la distorsione dalla camera """
def undistort(img,mtx,dist,newcameramtx,roi):
    img = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    img = img [y:y + h, x:x + w]
    return img