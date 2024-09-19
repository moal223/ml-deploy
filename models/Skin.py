import tensorflow as tf
import numpy as np

class SkinModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def preprocessing(self, image):

        image = image.resize((224, 224))  # Resize the image to the required size
        image = np.array(image)  # Convert the image to a numpy array
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image



    def predict(self, input_tensor):
        # Perform inference
        output_h5 = self.model.predict(input_tensor)
        output_list = output_h5.tolist()  # Convert NumPy array to Python list

        max_value = max(output_list[0])
        max_index = output_list[0].index(max_value)

        if max_value < 0.4:
            return "no"

        disease = {
            0: 'cellulitis',
            1: 'impetigo',
            2: 'athlete-foot',
            3: 'nail-fungus',
            4: 'ringworm',
            5: 'cutaneous-larva-migrans',
            6: 'chickenpox',
            7: 'shingles',
            8: 'no'
        }


        # Map the index to the wound type
        return disease.get(max_index, "Unknown Disease")
