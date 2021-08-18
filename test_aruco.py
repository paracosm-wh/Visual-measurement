import cv2 as cv
from cv2 import aruco

video_src = cv.VideoCapture(1)

if video_src.isOpened():
    print('video is opened')
while True:
    open_bool, frame = video_src.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    dictionary = aruco.Dictionary_get(aruco.DICT_5X5_100)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dictionary, parameters=parameters)
    aruco.drawDetectedMarkers(frame, corners, ids)

    cv.imshow("frame", frame)
    if cv.waitKey(10) & 0xFF == 27:
        break
video_src.release()
cv.destroyAllWindows()
