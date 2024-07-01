import os
import cv2
import numpy as np
import pytesseract
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
from sudoku_solver import solve_sudoku
from sudoku_image_processing import extract_sudoku_grid, recognize_digits, overlay_solution, order_points

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process the uploaded image
            warped = extract_sudoku_grid(filepath)
            sudoku_grid = recognize_digits(warped)
            solve_sudoku(sudoku_grid)
            solved_image = overlay_solution(warped, sudoku_grid)

            # Save the solved image
            solved_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'solved_' + filename)
            cv2.imwrite(solved_filepath, solved_image)

            return send_file(solved_filepath, mimetype='image/png')
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
