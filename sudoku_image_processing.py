import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pytesseract

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def extract_sudoku_grid(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Preprocess the image
    blurred = cv2.GaussianBlur(image, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours and extract the largest one
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    sudoku_contour = contours[0]

    # Approximate the contour to get a rectangle
    epsilon = 0.02 * cv2.arcLength(sudoku_contour, True)
    approx = cv2.approxPolyDP(sudoku_contour, epsilon, True)

    # Ensure we have exactly four points
    if len(approx) == 4:
        approx = order_points(np.squeeze(approx))

        # Warp the image to a square
        sudoku_size = 450
        output_points = np.array([[0, 0], [sudoku_size, 0], [sudoku_size, sudoku_size], [0, sudoku_size]], dtype='float32')
        matrix = cv2.getPerspectiveTransform(approx, output_points)
        warped = cv2.warpPerspective(image, matrix, (sudoku_size, sudoku_size))

        cv2.imshow("Warped Sudoku", warped)
        return warped
    else:
        print("Unable to find Sudoku grid.")


# Load a pre-trained digit recognition model (optional)
# model = load_model('digit_recognition_model.h5')

def recognize_digits(warped):
# Define cell size and extract each cell
    sudoku_size=450
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    cell_size = sudoku_size // 9
    sudoku_grid = np.zeros((9, 9), dtype=int)

    for i in range(9):
        for j in range(9):
            x = i * cell_size
            y = j * cell_size
            cell = warped[y:y + cell_size, x:x + cell_size]
            cv2.imshow("Cell", cell)
            digit = pytesseract.image_to_string(cell, config='--psm 10 digits')
            try:
                val=int(digit)  
                if(val>=10):
                    val=val/10
                sudoku_grid[j, i] = val
            except ValueError:
                sudoku_grid[j, i] = 0
    print(sudoku_grid)
    return sudoku_grid
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def overlay_solution(warped, sudoku_grid):
    cell_size = 50
    for i in range(9):
        for j in range(9):
            if sudoku_grid[j, i] != 0:
                x = i * cell_size
                y = j * cell_size
                cv2.putText(warped, str(sudoku_grid[j, i]), (x + 15, y + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return warped
