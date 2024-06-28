import cv2
import numpy as np

def preprocess_image(image_path):
    # Load the scanned image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Histogram Equalization
    equalized = cv2.equalizeHist(gray)

    # Step 2: Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        equalized, 
        200, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2
    )

    # Step 3: Shadow Removal
    dilated = cv2.dilate(adaptive_thresh, np.ones((5, 5), np.uint8))
    bg_img = cv2.medianBlur(dilated, 21)
    diff_img = 255 - cv2.absdiff(gray, bg_img)
    shadow_removed = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Step 4: Illumination Correction
    corrected = cv2.normalize(equalized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Step 5: Noise Reduction
    blurred = cv2.GaussianBlur(corrected, (5, 5), 0)
    binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Step 6: Edge Detection
    edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)

    # Step 7: Contour Finding
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour which should be the exam sheet
    contour = max(contours, key=cv2.contourArea)

    # Step 8: Perspective Transformation
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

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
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        # Destination points for the perspective transform
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")

        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # Convert the warped image to grayscale and binarize it
        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blurred_warped = cv2.GaussianBlur(gray_warped, (5, 5), 0)
        binary_warped = cv2.threshold(blurred_warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        return binary_warped
    else:
        print("Could not find the document edges. Ensure the exam sheet is fully visible in the image.")
        return None

# Load and preprocess the image
image_path = 'objective_sample.jpg'
processed_image = preprocess_image(image_path)

# Display the output
if processed_image is not None:
    cv2.imshow('Processed Image', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
