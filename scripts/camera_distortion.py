import cv2
import numpy as np
from ctu_crs import CRS97  # or CRS93
import matplotlib.pyplot as plt
import os

calibration_file = 'camera_calibration.npz'

def show_image(window_name, image, wait_time=0):
    # use matplotlib to show image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(window_name)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(wait_time)

def save_calibration():
    robot = CRS97()
    marker_size = 0.034  # size of one square in the checkerboard (in meters)
    marker_dimensions = (6, 4)  # number of inner corners per a chessboard row and column

    img_directory = "scripts/calibration_images/"
    image_files = [f for f in os.listdir(img_directory) if f.endswith('.png') or f.endswith('.jpg')]

    try:

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((marker_dimensions[0]*marker_dimensions[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:marker_dimensions[0],0:marker_dimensions[1]].T.reshape(-1,2)
        
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        for image_file in image_files:
            print(f"Processing image: {image_file}")
            img_path = os.path.join(img_directory, image_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image {img_path}")
                continue
            # img = cv2.imread("172825314-73366e8b-01d3-4f4e-8851-6f54a081b343.png")
            
            # Calculate and display distortion from checkeredboard pattern
            # https://stackoverflow.com/a/66232363

            # lwr = np.array([0, 0, 0])
            # upr = np.array([255, 255, 252])
            # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # msk = cv2.inRange(hsv, lwr, upr)

            # show_image('Mask', msk, wait_time=1)

            # krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
            # dlt = cv2.dilate(msk, krn, iterations=5)
            # res = 255 - cv2.bitwise_and(dlt, msk)
            # res = np.uint8(res)

            res = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cover the borders to avoid false detections
            h, w = res.shape
            border = 100
            res[0:border, :] = 255
            res[h-border:h, :] = 255
            res[:, 0:border] = 255
            res[:, w-border:w] = 255
            # show_image('Captured Image', res, wait_time=1)
        
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(res, marker_dimensions, flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                    cv2.CALIB_CB_FAST_CHECK +
                                                    cv2.CALIB_CB_NORMALIZE_IMAGE)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            print("found corners:", ret)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
        
                corners2 = cv2.cornerSubPix(res,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)
        
                # Draw and display the corners
                # cv2.drawChessboardCorners(img, marker_dimensions, corners2, ret)
                # show_image('img', img, wait_time=1)

        # Calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, res.shape[::-1], None, None)

        # undistort

        img = cv2.imread(os.path.join(img_directory, image_files[0]))
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

        # save mtx, dist, roi and newcameramtx to npy file
        np.savez(calibration_file, mtx=mtx, dist=dist, roi=roi, newcameramtx=newcameramtx)
        
        dst = undistort_image(img)
        cv2.imwrite('calibresult2.png', dst)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        plt.close('all')

def undistort_image(img):
    # load mtx, dist, roi and newcameramtx from npy file
    calib_data = np.load(calibration_file)
    mtx = calib_data['mtx']
    dist = calib_data['dist']
    newcameramtx = calib_data['newcameramtx']
    roi = calib_data['roi']

    # undistort
    h, w = img.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

if __name__ == "__main__":
    save_calibration()


