import tensorflow as tf
import numpy as np
from PIL import Image

class SkinDisease:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def preprocessing(self, image):
        # Ensure the image is a PIL image
        img_resized = image.resize((224, 224))
        img_scaled = img_resized / 255.0  # Scale pixel values to the range [0,1]
        img_scaled = np.expand_dims(img_scaled, axis=0)  # Add an extra dimension to make it (1, 224, 224, 3)
        return img_scaled

    def predict(self, input_tensor):
        try:
            # Perform inference
            prediction = self.model.predict(input_tensor)

            # Extract the predicted class
            predicted_label = np.argmax(prediction)

             # Print the result
            label_mapping = {0: 'cellulitis', 1: 'impetigo', 2: 'athlete-foot', 3: 'nail-fungus', 
                         4: 'ringworm', 5: 'cutaneous-larva-migrans', 6: 'chickenpox', 7: 'shingles'}
            return label_mapping[predicted_label]
        except Exception as e:
            return "error"

        