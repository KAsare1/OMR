from preprocessing import *
from tests.test import *


# answer_key = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'A', 7: 'B', 8: 'C', 9: 'D', 10: 'E', 
#               11: 'A', 12: 'B', 13: 'C', 14: 'D', 15: 'E', 16: 'A', 17: 'B', 18: 'C', 19: 'D', 20: 'E', 
#               21: 'A', 22: 'B', 23: 'C', 24: 'D', 25: 'E', 26: 'A', 27: 'B', 28: 'C', 29: 'D', 30: 'E', 
#               31: 'A', 32: 'B', 33: 'C', 34: 'D', 35: 'E', 36: 'A', 37: 'B', 38: 'C', 39: 'D', 40: 'E', 
#               41: 'A', 42: 'B', 4: 'C', 44: 'D', 45: 'E', 46: 'A', 47: 'B', 48: 'C', 49: 'D', 50: 'E', 
#               51: 'A', 52: 'B', 53: 'C', 54: 'D', 55: 'E', 56: 'A, 57: 'B', 58: 'C', 59: 'D', 60: 'E'}

# detected_answers, score = process_answer_sheet('objective_sample.jpg', answer_key)

preprocess_image('objective_sample.jpg')