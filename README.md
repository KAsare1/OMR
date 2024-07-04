# Automated Exam Paper Marking System

This project aims to build an automated exam paper marking system that uses a phone camera to scan and automatically mark multiple-choice exam papers. The project involves using OpenCV in Python to preprocess the image, detect and extract the answer sheet, and analyze the marked answers.

## Table of Contents

- [Project Description](#project-description)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Code Breakdown](#code-breakdown)
- [Contributors](#contributors)

## Project Description

This project leverages computer vision techniques to automate the grading of multiple-choice exams. It processes an image of the answer sheet, detects the contours, extracts the relevant section, and identifies the marked answers. The project uses OpenCV for image processing and analysis.

## Prerequisites

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- Pillow

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/automated-exam-paper-marking.git
   cd automated-exam-paper-marking
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place the image of the marked answer sheet in the `images` directory. Ensure the image name is `answer_sheet_original_marked_one.jpg`.
2. Run the main script:
   ```bash
   python main.py
   ```
3. The processed image and detected answers will be displayed.

## Code Breakdown

### Importing Libraries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
```

## Loading test image

```
image = cv2.imread('images/answer_sheet_original_marked_one.jpg')
```

## Converting to gray scale

```
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image_gray)
```

## Noise Reduction and Thresholding

```
image_noise_reduction = cv2.GaussianBlur(image_gray, (5,5), 1)
_, binary = cv2.threshold(image_noise_reduction, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
plt.imshow(binary)
```

## Edge Detection

```
image_edges = cv2.Canny(binary, 10, 50)
plt.imshow(image_edges)
```

## Finding and Drawing Contours

```
img_contours = image.copy()
contours, hierarchy = cv2.findContours(image_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 10)
plt.imshow(img_contours)
```

## Filtering and Sorting Contours

```
def rectangle_contour(contours):
    rect_contours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > 100:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rect_contours.append(i)
    rect_contours = sorted(rect_contours, key=cv2.contourArea, reverse=True)
    return rect_contours

rectCon = rectangle_contour(contours)
```

## Extracting Corner Points

```
def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
    return approx
```

## Reordering Corner Points

```
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(axis=1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew
```

## Perspective Transform and Image Warping

```
answered_objectives = getCornerPoints(rectCon[2])
img_answered = image.copy()

widthImg = 1200
heightImg = 1200

if answered_objectives.size != 0:
    cv2.drawContours(img_answered, answered_objectives, -1, (255, 255, 255), 50)
    answered_objectives = reorder(answered_objectives)
    pt1 = np.float32(answered_objectives)
    pt2 = np.float32([[0, 0], [widthImg, 0], [widthImg, heightImg], [0, heightImg]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgWarp = cv2.warpPerspective(img_answered, matrix, (widthImg, heightImg))
    imgFlipped = cv2.flip(imgWarp, 1)
    imgRotated = cv2.rotate(imgFlipped, cv2.ROTATE_90_COUNTERCLOCKWISE)

plt.imshow(imgRotated)
plt.axis('off')
plt.show()
```

## Thresholding Detected Answered Portions

```
detectedImage = cv2.cvtColor(imgRotated, cv2.COLOR_BGR2RGB)
detectedImage = cv2.cvtColor(imgRotated, cv2.COLOR_RGB2GRAY)
imageThreshold = cv2.threshold(detectedImage, 150, 255, cv2.THRESH_BINARY_INV)[1]
plt.imshow(imageThreshold)
```

## Splitting the Image into Sections

```
def get_section_coordinates(image_height, num_sections=4):
    section_height = (image_height // num_sections) - 5
    coordinates = [(i * section_height, (i + 1) * section_height) for i in range(num_sections)]
    print(coordinates)
    return coordinates
```

## Splitting Sections into Boxes

```
def splitBoxes(img, y_coordinates, splits_per_section=5):
    boxes = []
    for y1, y2 in y_coordinates:
        section = img[y1:y2, :]
        pad_size = (splits_per_section - section.shape[0] % splits_per_section) % splits_per_section
        section_padded = np.pad(section, ((0, pad_size), (0, 0)), mode='constant', constant_values=255)
        rows = np.vsplit(section_padded, splits_per_section)
        for row in rows:
            col_pad_size = (6 - row.shape[1] % 6) % 6
            row_padded = np.pad(row, ((0, 0), (0, col_pad_size)), mode='constant', constant_values=255)
            cols = np.hsplit(row_padded, 6)
            for box in cols:
                boxes.append(box)
    return boxes
```

## Calculating Pixel Values

```
image_height = imageThreshold.shape[0]
y_coordinates = get_section_coordinates(image_height, num_sections=4)
boxes = splitBoxes(imageThreshold, y_coordinates)

number_of_questions = 20
choices = 6
pixelValues = np.zeros((number_of_questions, choices))
countC = 0
countR = 0

for img in boxes:
    totalPixels = cv2.countNonZero(img)
    pixelValues[countR][countC] = totalPixels
    countC += 1
    if countC == choices:
        countR += 1
        countC = 0
print(pixelValues[0])
```

## Determining the Answer Index

```
myIndex = []
for x in range(number_of_questions):
    arr = pixelValues[x]
    myIndexVal = np.argmax(arr)
    myIndex.append(myIndexVal)

print(myIndex)
len(myIndex)
```

# Contributors

#### Kofi Asare-Amankwah

#### Noble Vulley

#### Prince Kongo
