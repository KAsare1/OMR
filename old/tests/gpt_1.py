import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(title, image, cmap='gray'):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()

# Load the first image
image_path = 'answer_sheet_one.jpg'
image = cv2.imread(image_path)

# Display original image
display_image("Original Image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display grayscale image
display_image("Grayscale Image", gray)

# Apply GaussianBlur to reduce noise and improve contour detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Display blurred image
display_image("Blurred Image", blurred)

# Apply adaptive thresholding
binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Display binary image
display_image("Binary Image", binary_image)

# Find contours
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Function to detect and warp the answer sheet to a bird's-eye view
def extract_exam_sheet(image, contours):
    if len(contours) == 0:
        return None
    
    # Find the largest rectangular contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            doc_cnts = approx
            break
    else:
        return None
    
    # Apply perspective transform to get a top-down view of the document
    rect = np.array([doc_cnts[0][0], doc_cnts[1][0], doc_cnts[2][0], doc_cnts[3][0]], dtype="float32")
    
    # Get top-down view dimensions
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

warped = extract_exam_sheet(image, contours)

# Display warped image
if warped is not None:
    display_image("Warped Image", cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))

# Updated positions based on the layout of your answer sheet
positions = [
    (135, 180, 160, 200),  # Question 1, Option A
    (170, 180, 195, 200),  # Question 1, Option B
    (205, 180, 230, 200),  # Question 1, Option C
    (240, 180, 265, 200),  # Question 1, Option D
    (275, 180, 300, 200),  # Question 1, Option E
    (135, 210, 160, 230),  # Question 2, Option A
    (170, 210, 195, 230),  # Question 2, Option B
    (205, 210, 230, 230),  # Question 2, Option C
    (240, 210, 265, 230),  # Question 2, Option D
    (275, 210, 300, 230),  # Question 2, Option E
    # Add similar tuples for each question option on the sheet
]

# Hypothetical answer key for illustration purposes
answer_key = {
    1: 'A',
    2: 'C',
    3: 'B',
    4: 'D',
    5: 'A',
    # Add the actual answer key here
}

def detect_shaded_answers(binary_warped, answer_key):
    # Initialize score and detected answers
    score = 0
    detected_answers = {}
    question_number = 1
    
    for pos in positions:
        x1, y1, x2, y2 = pos
        roi = binary_warped[y1:y2, x1:x2]
        total_pixels = roi.size
        if total_pixels == 0:
            continue
        non_zero_pixels = cv2.countNonZero(roi)
        shade_ratio = non_zero_pixels / total_pixels

        if shade_ratio > 0.5:
            detected_answer = 'A'  # Change this based on the position of the answer box
            correct_answer = answer_key.get(question_number)
            if detected_answer == correct_answer:
                score += 1
            
            detected_answers[question_number] = detected_answer
        else:
            detected_answers[question_number] = "Not Answered"

        question_number += 1
    
    print(f"Final Score: {score}/{len(answer_key)}")
    return detected_answers, score

# Step 6: Detect Shaded Answers and Calculate Score
if warped is not None:
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred_warped = cv2.GaussianBlur(gray_warped, (5, 5), 0)
    binary_warped = cv2.threshold(blurred_warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Display preprocessed warped images
    display_image("Grayscale Warped Image", gray_warped)
    display_image("Blurred Warped Image", blurred_warped)
    display_image("Binary Warped Image", binary_warped)

    detected_answers, score = detect_shaded_answers(binary_warped, answer_key)
    print(f"Detected Answers: {detected_answers}")
    print(f"Final Score: {score}/{len(answer_key)}")
else:
    print("Could not find the document edges. Ensure the exam sheet is fully visible in the image.")
