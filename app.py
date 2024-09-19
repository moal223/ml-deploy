from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from models.burnModel import BurnModel
import tensorflow as tf
from models.TypeModel import Type
from models.Skin import SkinModel


app = Flask(__name__)

# Load TensorFlow models
burnDegree = BurnModel('models/purnDegree.h5')
#typeModel = Type('models/Type.h5')
skinModel = SkinModel('models/skin_disease_model.h5')



def preprocess_image(image):
    """Preprocess the image to fit the model's input requirements."""
    image = image.resize((224, 224))  # Resize the image to the required size
    image = np.array(image)  # Convert the image to a numpy array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image




@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files: 
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        response_list = []
        # Open the image file
        image = Image.open(file.stream)
        
        image_process = preprocess_image(image)
        response = burnDegree.predict_burn_degree(image_process)
        response_list.append(response)

        image_skin = skinModel.preprocessing(image)
        skin_response = skinModel.predict(image_skin)
        response_list.append(skin_response)
        
        # Return the prediction result as JSON
        return jsonify({'Output': response_list})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)