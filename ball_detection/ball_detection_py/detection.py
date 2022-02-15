import cv2
import numpy as np
import imutils


def nothing(x):
    pass


cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    #frame = cv2.imread('ball.jpg')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_b = np.array([26, 29, 128])
    u_b = np.array([87, 255, 255])

    mask = cv2.inRange(hsv, l_b, u_b)

    res = cv2.bitwise_and(frame, frame, mask=mask)
    mask = cv2.bitwise_not(mask, mask)
    params = cv2.SimpleBlobDetector_Params()

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 50000

#     # Filter by Circularity
#     params.filterByCircularity = True
#     params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
#     params.filterByInertia = True
#     params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    # OLD: detector = cv2.SimpleBlobDetector(params)
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(mask)

    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array(
        []), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    im_with_keypoints = imutils.resize(im_with_keypoints, width=400)
    cv2.imshow("Keypoints", im_with_keypoints)

    frame = imutils.resize(frame, width=400)
    mask = imutils.resize(mask, width=400)
    res = imutils.resize(res, width=400)
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)
    print(keypoints)
    for keyPoint in keypoints:
        x = keyPoint.pt[1]
        y = keyPoint.pt[0]
        s = keyPoint.size
        print(x, y, s)
        print(im_with_keypoints.shape)

    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()
