import cv2
import numpy as np
import os

file = "./test.jpg"




def main():
    image = cv2.imread(file)
    image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(image_gray, 100, 0.01, 10)
    corners = np.int8(corners)

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image_gray, (x, y), 5, (0, 255, 0), 1)

    cv2.imshow("Original", image_gray)




    cv2.waitKey(0)
    cv2.destroyAllWindows()
    






if __name__ == "__main__":
    main()