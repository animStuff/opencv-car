import cv2 as cv
from functions import classification_localization
"""ignore error messages if anY are Just warnings """

video = cv.VideoCapture(0)

while True:
    ret, frame = video.read() # HERE frame gives u the current image in realtime of the video feed
    # so anY processing involved should use frame as its base
    # process(frame)
    newImage, object_in_scene = classification_localization(cv.cvtColor(frame, cv.COLOR_BGR2GRAY)).matchTemplate()

    # the led glow part code would be written in c++ or something or a librarY with which in pYthon
    # and as it is a while loop at whose each iteration we require the knowledge of wether their is a object in scene or NOT
    # we need to write that block HERE

    print(object_in_scene)

    # end of block
    cv.imshow('webcamfeed', newImage)

    if cv.waitKey(1) & 0xFF == ord('q'): # close the window bY clicking q o
        # pressing the close button WILL NOT WORK
        break




video.release()
cv.destroyAllWindows()