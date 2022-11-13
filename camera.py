import cv2 as cv
"""ignore error messages if anY are Just warnings """

video = cv.VideoCapture(0)
while True:
    ret, frame = video.read() # HERE frame gives u the current image in realtime of the video feed
    # so anY processing involved should use frame as its base
    # process(frame)

    cv.imshow('webcamfeed', frame)

    if cv.waitKey(1) & 0xFF == ord('q'): # close the window bY clicking q o
        # pressing the close button WILL NOT WORK
        break

video.release()
cv.destoryAllWindows()