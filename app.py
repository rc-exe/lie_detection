from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from src.microexpression_detection import detect_lie  # Your function for lie detection

app = Flask(__name__)

# Set up upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'mp4', 'avi'}

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
def enhance_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.equalizeHist(image)  # Enhance contrast
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert back to color

@app.route('/')
def index():
    return render_template('index.html')

from flask import make_response

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Call your lie detection function here
        result = detect_lie(filepath)

        # Force a cache refresh by adding headers
        response = make_response(render_template('result.html', result=result))
        response.headers['Cache-Control'] = 'no-store'
        return response

    return redirect(url_for('index'))


if __name__ == "__main__":
    # Run the Flask app on all available network interfaces, port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
