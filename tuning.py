import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    # Load the scanned image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(10, 10))
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.show()

    # Apply a threshold to get a binary image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    plt.figure(figsize=(10, 10))
    plt.imshow(thresh, cmap='gray')
    plt.title('Thresholded Image')
    plt.show()

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the answer sheet
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    answer_sheet_contour = contours[0]

    # Approximate the contour to get a rectangle
    epsilon = 0.02 * cv2.arcLength(answer_sheet_contour, True)
    approx = cv2.approxPolyDP(answer_sheet_contour, epsilon, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        # Order points in clockwise order: top-left, top-right, bottom-right, bottom-left
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect

        # Compute the width and height of the new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Destination points for the perspective transform
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        plt.title('Warped Image')
        plt.show()

        # Convert the warped image to grayscale and binarize it
        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        plt.figure(figsize=(10, 10))
        plt.imshow(gray_warped, cmap='gray')
        plt.title('Warped Grayscale Image')
        plt.show()

        _, binary_warped = cv2.threshold(gray_warped, 150, 255, cv2.THRESH_BINARY_INV)
        plt.figure(figsize=(10, 10))
        plt.imshow(binary_warped, cmap='gray')
        plt.title('Binary Warped Image')
        plt.show()

        return binary_warped
    else:
        print("Could not find the document edges. Ensure the exam sheet is fully visible in the image.")
        return None

# Example usage
image_path = 'objective_sample.jpg'
processed_image = preprocess_image(image_path)

# Display the processed image
if processed_image is not None:
    cv2.imshow('Processed Image', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
