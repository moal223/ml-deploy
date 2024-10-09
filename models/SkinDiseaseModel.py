import tensorflow as tf
import numpy as np
from PIL import Image

class SkinDisease:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def preprocessing(self, image):
        # Ensure the image is a PIL image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Resize the image with appropriate parameters
        resized_img = image.resize((224, 224), Image.Resampling.NEAREST)

        # Convert to a NumPy array
        resized_img_np = np.array(resized_img)

        # Expand dimensions to match model's input shape
        resized_img_np = np.expand_dims(resized_img_np, axis=0)
        return resized_img_np

    def predict(self, input_tensor):
        # Perform inference
        output_h5 = self.model.predict(input_tensor)
        output_list = output_h5.tolist()  # Convert NumPy array to Python list

        # Find the index of the highest value
        max_value = max(output_list[0])
        max_index = output_list[0].index(max_value)

        disease_train_label_dic={
            0: 'cellulitis',
            1: 'impetigo',
            2: 'athlete-foot',
            3: 'nail-fungus',
            4: 'ringworm',
            5: 'cutaneous-larva-migrans',
            6: 'chickenpox',
            7: 'shingles',
            8:  'normal',
        }

        # Return the index of the highest value
        return disease_train_label_dic[max_index]