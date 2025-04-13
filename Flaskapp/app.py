import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


model = load_model(r'Flaskapp\vgg16_binary_classifier.h5')


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_path = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)


            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0


            
            pred = model.predict(img_array)[0][0]
            confidence = round(float(pred) * 100, 2)
            print(confidence)
            if pred >= 0.6:
                prediction = f"Defected ({confidence}%)"
            elif pred < 0.4:
                prediction = f"Not Defected ({100 - confidence}%)"
            else:
                prediction = f"Uncertain ({confidence}%)"


    return render_template("index.html", prediction=prediction, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=False,port=5001)
