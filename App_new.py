from flask import Flask, render_template, request
import mysql.connector
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model
model = tf.keras.models.load_model("best_model (8).keras")

# Connect to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="anu@sreelu36",
    database="plant_disease_db"
)

cursor = db.cursor()

# Disease labels (must match model training order)
class_names = ['Pepper__bell___Bacterial_spot',
'Pepper__bell___healthy',
'Potato___Early_blight',
'Potato___Late_blight',
'Potato___healthy',
'Tomato_Bacterial_spot',
'Tomato_Early_blight',
'Tomato_Late_blight',
'Tomato_Leaf_Mold',
'Tomato_Septoria_leaf_spot',
'Tomato_Spider_mites_Two_spotted_spider_mite',
'Tomato__Target_Spot',
'Tomato__Tomato_YellowLeaf__Curl_Virus',
'Tomato__Tomato_mosaic_virus',
'Tomato_healthy']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/upload')
def upload():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess image
    img = Image.open(filepath).resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]

    # Fetch pesticide from MySQL
    query = """
    SELECT p.pesticide_name, p.usage_instructions
    FROM disease d
    JOIN recommendation r ON d.disease_id = r.disease_id
    JOIN pesticide p ON r.pesticide_id = p.pesticide_id
    WHERE d.disease_name = %s
    """

    cursor.execute(query, (predicted_class,))
    result = cursor.fetchone()

    pesticide = result[0]
    usage = result[1]

    return render_template("result.html",
                           disease=predicted_class,
                           pesticide=pesticide,
                           usage=usage,
                           image=filepath)

if __name__ == "__main__":
    app.run(debug=True)