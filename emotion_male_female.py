import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import base64
from mtcnn import MTCNN

app = Flask(__name__)

# Set the path to the upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained models
gender_model = load_model(r"C:\python\projects\malefemale\male_female.h5")
emotion_model = load_model(r"C:\python\projects\emotion\emotion.h5")

# Initialize global variables


# Function to detect faces in an image using MTCNN
def detect_faces_mtcnn(image):
    detector = MTCNN()
    results = detector.detect_faces(image)
    faces = []
    for result in results:
        x, y, w, h = result['box']
        faces.append((x, y, w, h))
    return faces

# Function to preprocess image for gender classification
def preprocess_gender_image(img):
    img = cv2.resize(img, (120, 120))  # Resize the image to match the model's expected sizing
    img = np.reshape(img, [1, 120, 120, 3])  # Reshape image to match model's expected sizing
    img = img.astype('float32')
    img /= 255  # Normalize the pixel values to be between 0 and 1
    return img

# Function to preprocess image for emotion detection
def preprocess_emotion_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for emotion detection
    img = cv2.resize(img, (128, 128))  # Resize the image to match the model's expected sizing
    img = np.stack((img,) * 3, axis=-1)  # Stack the grayscale image to simulate RGB channels
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.astype('float32')
    img /= 255  # Normalize the pixel values to be between 0 and 1
    return img


# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Ensure the uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read the uploaded image
        input_image = cv2.imread(filepath)

        # Detect faces in the input image using MTCNN
        detected_faces = detect_faces_mtcnn(input_image)
        male_count=0
        female_count=0
        emotion_names=[]
        # Perform gender classification and emotion detection for each detected face
        for (x, y, w, h) in detected_faces:
            face = input_image[y:y+h, x:x+w]
            
            # Gender classification
            gender_face = preprocess_gender_image(face)
            gender_prediction = gender_model.predict(gender_face)
            gender_class_index = np.argmax(gender_prediction)
            gender_class_names = ["female", "male"]
            gender_predicted_class = gender_class_names[gender_class_index]
            
            # Emotion detection
            emotion_face = preprocess_emotion_image(face)
            emotion_prediction = emotion_model.predict(emotion_face)
            emotion_class_index = np.argmax(emotion_prediction)
            emotion_class_names = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            emotion_predicted_class = emotion_class_names[emotion_class_index]
            emotion_names.append(emotion_predicted_class)
            # Update counts based on gender
            
            if gender_predicted_class == 'female':
                female_count += 1
            else:
                male_count += 1

            # Display predicted gender and emotion on the image
            # cv2.putText(input_image, f"{gender_predicted_class}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(input_image, f" {emotion_predicted_class}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Save the annotated image with predictions
        annotated_filename = f"annotated_{filename}"
        annotated_filepath = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
        cv2.imwrite(annotated_filepath, input_image)

        # Encode the annotated image as Base64
        with open(annotated_filepath, "rb") as img_file:
            annotated_image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # Return the Base64 encoded image and results to the client for display
        return render_template('index.html', result_image=annotated_image_base64, male_count=male_count, female_count=female_count,emotion_names=emotion_names)

   

if __name__ == '__main__':
    a=input()
    print(a)
    app.run(debug=True)
