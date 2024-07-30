import cv2
import numpy as np


def preprocess(imgPath):
    image = cv2.imread(imgPath)
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageNoiseReduction = cv2.GaussianBlur(imageGray, (5,5), 1)
    _, binary = cv2.threshold(imageNoiseReduction, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    imageEdges = cv2.Canny(binary, 10, 50)
    img_contours = image.copy()
    contours, hierarchy = cv2.findContours(imageEdges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 10)
    return contours, hierarchy, image


def rectangle_contour(imgPath):
    contours,hierarchy, image = preprocess(imgPath)
    rect_contours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area>100:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            if len(approx) == 4:
                rect_contours.append(i)

    rect_contours = sorted(rect_contours, key=cv2.contourArea, reverse=True)
    return rect_contours


def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02*peri, True)
    
    return approx


def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2), np.int32)
    add = myPoints.sum(axis=1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmin(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmin(diff)]

    return myPointsNew


def region_of_interest(imgPath):
     contour, hierarchy, image = preprocess(imgPath)
     img_answered = image.copy()
     widthImg = 1200
     heightImg =  1200
     captured_regions = []
     rectCon = rectangle_contour(imgPath)
     for i in range(3):
        answered_objectives = getCornerPoints(rectCon[i])
        if answered_objectives.size != 0:
            cv2.drawContours(img_answered, answered_objectives, -1, (255, 255, 255), 50)
            reorder(answered_objectives)

            pt1 = np.float32(answered_objectives)
            pt2 = np.float32([[0,0], [widthImg, 0], [widthImg, heightImg], [0,heightImg]])

            matrix = cv2.getPerspectiveTransform(pt1, pt2)
            imgWarp = cv2.warpPerspective(img_answered, matrix, (widthImg, heightImg))
            imgFlipped = cv2.flip(imgWarp, 1)
            imgRotated = cv2.rotate(imgFlipped, cv2.ROTATE_90_COUNTERCLOCKWISE)
            captured_regions.append(imgRotated)
     return captured_regions     


def thresholding(imgPath):
    thresholded_images =[]
    imageThreshold = []
    for image in region_of_interest(imgPath):
        detectedImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detectedImage = cv2.cvtColor(detectedImage, cv2.COLOR_RGB2GRAY)
        imageThreshold = cv2.threshold(detectedImage, 150, 255, cv2.THRESH_BINARY_INV)[1]
        thresholded_images.append(imageThreshold)
    return thresholded_images, imageThreshold    


def get_section_coordinates(image_height, num_sections=4):
    section_height = (image_height // num_sections)-5
    coordinates = [(i * section_height, (i + 1) * section_height) for i in range(num_sections)]
    print(coordinates)
    return coordinates


def splitBoxes(img, y_coordinates, splits_per_section=5):
    boxes = []
    for y1, y2 in y_coordinates:
        section = img[y1:y2, :]
        # Ensure the number of rows in the section is divisible by splits_per_section
        pad_size = (splits_per_section - section.shape[0] % splits_per_section) % splits_per_section
        section_padded = np.pad(section, ((0, pad_size), (0, 0)), mode='constant', constant_values=255)

        rows = np.vsplit(section_padded, splits_per_section)
        for row in rows:
            # Ensure the number of columns in the row is divisible by 6
            col_pad_size = (6 - row.shape[1] % 6) % 6
            row_padded = np.pad(row, ((0, 0), (0, col_pad_size)), mode='constant', constant_values=255)
            cols = np.hsplit(row_padded, 6)
            for box in cols:
                boxes.append(box)
    return boxes


def region_identification(img):
    non_zero_pixels = cv2.countNonZero(img)
    if non_zero_pixels > 2200:
        return "41"
    elif 2000 < non_zero_pixels <= 2200:  
        return "21"
    elif non_zero_pixels <= 2000:  
        return "1"
    else:
        return "Unknown"
    

def dict_image(imgPath):
    thresholded_images, imageThreshold = thresholding(imgPath)
    image_height = imageThreshold.shape[0]
    y_coordinates = get_section_coordinates(image_height, num_sections=4)
    first_image_dict = {}
    for image in thresholded_images:
        boxes = splitBoxes(image, y_coordinates)
        first_image_key = boxes[0]
        regionId = region_identification(first_image_key)
        first_image_dict[regionId] = boxes
    return first_image_dict    


def marking(boxes, number_of_questions=20, choices=6):
    pixel_values = np.zeros((number_of_questions, choices))
    count_c = 0
    count_r = 0

    for img in boxes:
        total_pixels = cv2.countNonZero(img)
        pixel_values[count_r][count_c] = total_pixels
        count_c += 1
        if count_c == choices:
            count_r += 1
            count_c = 0
        my_index = []
        
    for x in range(number_of_questions):
        arr = pixel_values[x]
        my_index_val = np.argmax(arr)  # Find the index of the maximum value in the row
        my_index.append(my_index_val)

    return my_index



def marker(imagePath):
    image_dict = dict_image(imagePath)
    marked_dict = {k: marking(v) for k, v in image_dict.items()}
    
    # Ensure the dictionary is JSON serializable
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {make_serializable(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set)):
            return [make_serializable(i) for i in obj]
        return obj

    serializable_dict = make_serializable(marked_dict)
    
    return serializable_dict



def compare_grades(dict1, dict2):
    total_grade = 0
    max_grade = 60
    
    # Ensure both dictionaries have the same keys
    if dict1.keys() != dict2.keys():
        return "Dictionaries have different keys"
    
    # Helper function to ensure the grades are in native Python types
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {make_serializable(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set)):
            return [make_serializable(i) for i in obj]
        return obj
    
    dict1 = make_serializable(dict1)
    dict2 = make_serializable(dict2)
    
    # Iterate through each key
    for key in dict1.keys():
        grades1 = dict1[key]
        grades2 = dict2[key]
        
        # Ensure both lists have the same length
        if len(grades1) != len(grades2):
            return "Grade lists have different lengths for key: {}".format(key)
        
        # Compare grades at each index
        for i in range(len(grades1)):
            if grades1[i] == grades2[i]:
                total_grade += grades1[i]
    
    return min(total_grade, max_grade)



    


