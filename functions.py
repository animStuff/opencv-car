import cv2, numpy as np
import os, copy

class LaneDetection:
    def __init__(self):
        self.ignore_mask_color = 255
        self.rho = 1
        self.maxLineLength = 20
        self.theta = np.pi/180 # equal to one degree
        self.maxLineGap = 100
        self.threshold = 10


    def masking(self, image) -> 'capturing':
        """only capturing the reuired white and Yellow lanes to filter out the useless information
        of the images that wouldnt be required during processing"""
        self.HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        self.lower_limits = [np.uint8([20, 100, 100]), np.uint8([0, 100, 100])] #Yellow, White
        self.upper_limits = [np.uint8([40, 255, 255]), np.uint8([10, 255, 255])]

        self.Wmask = cv2.inRange(self.HSV_image, self.lower_limits[1], self.upper_limits[1])
        self.Ymask = cv2.inRange(self.HSV_image, self.lower_limits[0], self.upper_limits[0])

        self.combinedMask = cv2.bitwise_or(self.Wmask, self.Ymask)
        self.maskedImage = cv2.bitwise_and(image, image, self.combinedMask)

        return self.maskedImage


    def edge_detection(self, image) -> 'CANNY ALGORITHM':
        """all operations noiseReduction, Non-maximum-suppresion etc are performed
        within cv2.canny it basically creates a binary image where the edges are highlighted
        in white.."""

        self.graY = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.edges = cv2.Canny(self.graY, 100, 100)

        return self.edges

    def ROI(self, image) -> 'extracting the info required':
        """there is a certain place or area from which we are always
        going to be taking information, (RESTRICTED environment) so we extract that part of the image
        so only those edges are present"""
        self.mask = np.zeros_like(image)

        # this block contains the extraction of the required roads (considering a standard camera pos)
        # neecs to be adjusted to suit our environment
        rows, cols = image.shape[:2]
        bottom_left  = [cols * 0.1, rows * 0.95]
        top_left     = [cols * 0.4, rows * 0.6]
        bottom_right = [cols * 0.9, rows * 0.95]
        top_right    = [cols * 0.6, rows * 0.6]

        self.vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        cv2.fillPoly(self.mask, self.vertices, self.ignore_mask_color)
        return cv2.bitwise_and(image, self.mask) # returning black pixels and the extracted region

    def HoughTransform(self, image) -> 'get lane lines':
        """https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html gets us the lines from the edges"""
        return cv2.HoughLinesP(image, rho = self.rho, theta = self.theta, threshold = self.threshold,
                           minLineLength = self.maxLineLength, maxLineGap = self.maxLineGap)

    def gaussianBlur(self, image) -> 'smoothing the image':
        """a bell curve but with z axis filter that smooths out the image by highlighting the
        mid region of the (5, 5) section during convolution"""
        return cv2.GaussianBlur(image, (5, 5), 0)

    def graYscale(self, image) -> 'onlY one channel':
        """converting from hsv->bgr-:GRAY"""

        return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)

class DrawLines:
    def __init__(self, image, hough):
        self.lLines, self.rLines = [], [] # (slope, intercept, None)
        self.lWeight, self.rWeight = [], [] # (1, length, None)
        self.image = image
        self.color = [255, 0, 0] # colors
        self.thickness = 12 # thickness
        self.hough = hough


    def draw_lanes(self):
        """to draw to the image"""
        template, lines = np.zeros_like(self.image), self.lengthLanes(self.hough)
        for line in lines:
            if line is not None:
                cv2.line(template, *line, self.color, self.thickness)

        return cv2.addWeighted(self.image, 1, template, 1, 0)


    def slope_intercept(self, lines) -> 'calculates slope and intercept of lines':
        """averaging the slopes and intercepts of the coordinates"""
        for line in lines:
            for X1, Y1, X2, Y2 in line:
                if X1 == X2:
                    continue

                self.slope = (Y2 - Y1)/(X2 - X1)
                self.intercept = Y1 - self.slope * X1
                self.length = np.sqrt((Y2 - Y1) ** 2 + (X2 - X1) ** 2)

                if self.slope < 0:
                    self.lLines.append((self.slope, self.intercept))
                    self.lWeight.append((self.length))

                else:
                    self.rLines.append((self.slope, self.intercept))
                    self.rWeight.append((self.length))

        self.lLane = np.dot(self.lWeight, self.lLines)/np.sum(self.lWeight) if len(self.lWeight) else None
        self.rLane = np.dot(self.rWeight, self.rLines)/np.sum(self.rWeight) if len(self.rWeight) else None

        return self.lLane, self.rLane # left, write (slope, intercept) averaged over all lines

    def pixelPoints(self, Y1, Y2, line) -> 'slope_intercept to pixel points':
        """conversion of 2d vector space points into pixel points"""
        if line is None:
            return None

        slope, intercept = line
        self.X1, self.X2 = int((Y1 - intercept)/slope), int((Y2 - intercept)/slope)
        self.Y1, self.Y2 = int(Y1), int(Y2)

        return ((self.X1, self.Y1), (self.X2, self.Y2))

    def lengthLanes(self, lines):
        """translation of all functions into one"""
        laneL, laneR = self.slope_intercept(lines)
        Y1 = self.image.shape[0]
        Y2 = 0.6 * Y1 #.6 multiplied as of region selection

        self.lineL = self.pixelPoints(Y1, Y2, laneL)
        self.lineR = self.pixelPoints(Y1, Y2, laneR)

        return self.lineL, self.lineR

class classification_localization:
    def __init__(self, imagePath):
        self.image = cv2.imread(imagePath, 0)
        self.templates = [(cv2.imread(f'classes/{fileName}', 0), fileName) for fileName in os.listdir('classes')]
        self.required = [(each, each.shape[0], each.shape[1], fileName) for each, fileName in self.templates]
        self.editedImage = cv2.imread(imagePath)

    def matchTemplate(self):
        for image, h, w, fileName in self.required:
            res, thresh = cv2.matchTemplate(self.image, image, cv2.TM_CCOEFF_NORMED), .8
            loc = np.where(res >= thresh)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(self.editedImage, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1) #BGR format
                #cv2.putText(self.editedImage, fileName, (pt[0], pt[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return self.editedImage
