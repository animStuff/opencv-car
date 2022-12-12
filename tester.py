from functions import LaneDetection, DrawLines, classification_localization
import os, cv2 as cv

if __name__ == '__main__':
    lanedetect = LaneDetection()
    for index, path in enumerate([f'test/roads/{imgName}' for imgName in os.listdir('test/roads')]):
        """remember to clear result dir before resuse"""
        image = cv.imread(path)
        HSV = lanedetect.masking(image)
        GRAY = lanedetect.graYscale(HSV)
        BLUR = lanedetect.gaussianBlur(GRAY)
        EDGE = lanedetect.edge_detection(BLUR)
        REGION = lanedetect.ROI(EDGE)
        HOUGH = lanedetect.HoughTransform(REGION)
        lanes_image = DrawLines(image, HOUGH).draw_lanes()
        cv.imwrite(f'results/roads/result{index}.jpg', lanes_image)

    for index, path in enumerate(f'test/localization/{imgName}' for imgName in os.listdir('test/localization')):

        newImage, _ = classification_localization(cv.imread(path, 0)).matchTemplate()
        cv.imwrite(f'results/local/result{index}.jpg', newImage)





