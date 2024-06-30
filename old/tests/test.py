import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_answer_sheet(image_path, answer_key):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Unable to read image file {image_path}")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to get a binary image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the answer sheet
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    answer_sheet_contour = contours[0]

    # Approximate the contour to get a rectangle
    epsilon = 0.02 * cv2.arcLength(answer_sheet_contour, True)
    approx = cv2.approxPolyDP(answer_sheet_contour, epsilon, True)

    # If the approximated contour has four points, we can assume it is the answer sheet
    if len(approx) == 4:
        pts = approx.reshape(4, 2)

        # Order points to [top-left, top-right, bottom-right, bottom-left]
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # Process the warped image to extract answers
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, warped_thresh = cv2.threshold(warped_gray, 150, 255, cv2.THRESH_BINARY_INV)

        # Display the preprocessed image
        plt.figure(figsize=(10, 10))
        plt.imshow(warped_thresh, cmap='gray')
        plt.title('Preprocessed Answer Sheet')
        plt.show()

        # Example: Extract answers from the first row
        row_height = maxHeight // 60  # Assuming 60 questions
        col_width = maxWidth // 5  # Assuming 5 options per question

        detected_answers = []
        for question_num in range(60):  # Loop through each question
            row_start = question_num * row_height
            row = warped_thresh[row_start:row_start + row_height, :]

            question_answers = []
            for i in range(5):
                col_start = i * col_width
                col = row[:, col_start:col_start + col_width]
                total = cv2.countNonZero(col)
                if total > 1000:  # Threshold to consider an answer as filled
                    question_answers.append(chr(65 + i))  # Convert index to letter

            if question_answers:
                detected_answers.append(question_answers[0])  # Assume only one answer can be marked

        print("Detected Answers:", detected_answers)

        # Compare with answer key
        score = sum(1 for i in range(len(detected_answers)) if detected_answers[i] == answer_key.get(i + 1))
        print("Score:", score)
        return detected_answers, score
    else:
        print("Answer sheet not detected properly.")
        return None


