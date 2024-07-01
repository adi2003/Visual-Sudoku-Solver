# Visual-Sudoku-Solver
A service that solves a sudoku configuration using the image of the sudoku grid.
This project is a Visual Sudoku Solver that allows users to upload an image of a Sudoku puzzle, recognizes the digits on the grid, solves the puzzle, and displays the solution over the original image. The project is implemented using Flask for the web interface, OpenCV for image processing, and a pytesseract for digit recognition.

Features
1. Upload an image of a Sudoku puzzle.
2. Automatically detect and extract the Sudoku grid from the image.
3. Recognize digits using a pre-trained neural network.
4. Solve the Sudoku puzzle.
5. Display the solved puzzle overlaid on the original image.

Prerequisites
1. Python 3.x
2. Flask
3. OpenCV
5. TensorFlow/Keras
6. Tesseract OCR

Setup Instructions
Clone the repository:

sh
Copy code
git clone https://github.com/adi2003/visual-sudoku-solver.git
cd visual-sudoku-solver
Create and activate a virtual environment:

sh
Copy code
python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
Install the required packages:

sh
Copy code
pip install -r requirements.txt

Place the background image in the static folder:

Create a static directory in your project root if it doesn't exist.
Move your background image (bgimage.jpg) into the static directory.
Run the Flask application:

sh
Copy code
python app.py
Open your web browser and go to:

sh
Copy code
http://127.0.0.1:5000/

Application Structure:

app.py: The main Flask application file.

templates/index.html: The main HTML file for the web interface.

static/bgimage.jpg: The background image for the web page.

Workflow

Uploading an Image:

Users can upload an image of a Sudoku puzzle through the web interface.

Image Processing:

The uploaded image is processed to detect and extract the Sudoku grid using OpenCV.The grid is transformed to a bird's-eye view perspective for better accuracy in digit recognition.

Digit Recognition:

Each cell in the Sudoku grid is preprocessed and fed into pytesseract pre-trained model to recognize the digits. If a digit is returned by the pytesseract api then we keep it else we allot it to 0 and assume that it was empty. The recognized digit is added to the grid.

Solving the Sudoku:

The recognized Sudoku grid is solved using a backtracking algorithm.

Overlaying the Solution:

The solved digits are overlaid onto the original image.
The final image, showing the solved Sudoku puzzle, is displayed to the user.

Example Usage

1. Upload an Image:<br>
 -- Click on the "Choose File" button and select an image of a Sudoku puzzle from your computer.<br>
 -- Click on "Upload and Solve".<br>
 
2. View Solved Puzzle:<br>
 -- The solved puzzle will be displayed with the solution overlaid on the original image.<br>
 -- You can download the solved image.

Troubleshooting: <br>
-- Ensure all required packages are installed.<br>
-- Verify the pre-trained MNIST model path is correct.<br>
-- Ensure Tesseract OCR is installed and configured correctly if you choose to use it.<br>
-- Check the Flask server logs for any errors.

Future Improvements: <br>
1. Improve digit recognition accuracy.<br>
2. Add more robust error handling and user feedback.<br>
3. Extend the application to handle more complex puzzles and features.
   
