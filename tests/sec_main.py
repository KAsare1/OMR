import cv2
import numpy as np
import matplotlib.pyplot as plt
from tests.process_two import *

def process_answers(binary_image, answer_key, fill_threshold=1000):
    height, width = binary_image.shape
    row_height = height // 60  # Assuming 60 questions
    col_width = width // 5  # Assuming 5 options per question

    detected_answers = []
    for question_num in range(60):  # Loop through each question
        row_start = question_num * row_height
        row = binary_image[row_start:row_start + row_height, :]

        question_answers = []
        for i in range(5):
            col_start = i * col_width
            col = row[:, col_start:col_start + col_width]
            total = cv2.countNonZero(col)

            if total > fill_threshold:  # Adjust this threshold as needed
                question_answers.append(chr(65 + i))  # Convert index to letter

        if not question_answers:
            detected_answers.append("Unanswered")
        else:
            detected_answers.append(question_answers[0])  # Assume only one answer can be marked

    print("Detected Answers:", detected_answers)

    # Compare with answer key and handle unanswered questions
    score = 0
    for i in range(len(detected_answers)):
        if detected_answers[i] != "Unanswered" and detected_answers[i] == answer_key.get(i + 1):
            score += 1

    print("Score:", score)
    return detected_answers, score

# Example usage
image_path = 'objective_sample.jpg'
answer_key = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 
              6: 'A', 7: 'B', 8: 'C', 9: 'D', 10: 'E', 
              11: 'A', 12: 'B', 13: 'C', 14: 'D', 15: 'E', 
              16: 'A', 17: 'B', 18: 'C', 19: 'D', 20: 'E', 
              21: 'A', 22: 'B', 23: 'C', 24: 'D', 25: 'E', 
              26: 'A', 27: 'B', 28: 'C', 29: 'D', 30: 'E', 
              31: 'A', 32: 'B', 33: 'C', 34: 'D', 35: 'E', 
              36: 'A', 37: 'B', 38: 'C', 39: 'D', 40: 'E', 
              41: 'A', 42: 'B', 43: 'C', 44: 'D', 45: 'E', 
              46: 'A', 47: 'B', 48: 'C', 49: 'D', 50: 'E', 
              51: 'A', 52: 'B', 53: 'C', 54: 'D', 55: 'E', 
              56: 'A', 57: 'B', 58: 'C', 59: 'D', 60: 'E'}

processed_image = preprocess_image(image_path)

# Display the processed image
if processed_image is not None:
    cv2.imshow('Processed Image', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Process answers and calculate the score
    detected_answers, score = process_answers(processed_image, answer_key, fill_threshold=1000)
