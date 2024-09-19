import tensorflow as tf
import numpy as np

class Type:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def preprocessing(self, image):
        # Convert to NumPy array if needed
        image_np = np.array(image)

        # Resize image using TensorFlow
        resized_img = self.resize_image(image_np, (224, 224))

        # Data Augmentation
        augmented_img = self.augment_image(resized_img)

        # Normalize image
        normalized_img = self.normalize_image(augmented_img)

        # Expand dimensions to fit model input
        processed_img = np.expand_dims(normalized_img, axis=0)

        return processed_img

    def resize_image(self, image, target_size):
        # Resize the image using TensorFlow
        image_resized = tf.image.resize(image, target_size)
        return image_resized.numpy()

    def augment_image(self, image):
        # Convert image to tensor
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

        # Random horizontal flip
        image_tensor = tf.image.random_flip_left_right(image_tensor)

        # Random vertical flip
        image_tensor = tf.image.random_flip_up_down(image_tensor)

        # Random rotation (rotating by 90, 180, or 270 degrees)
        image_tensor = tf.image.rot90(image_tensor, k=np.random.randint(0, 4))

        # Random contrast
        image_tensor = tf.image.random_contrast(image_tensor, 0.7, 1.3)

        # Random brightness
        image_tensor = tf.image.random_brightness(image_tensor, 0.2)

        return image_tensor.numpy()

    def normalize_image(self, image):
        # Assuming ImageNet mean and std
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        # Normalize image
        image = (image / 255.0 - imagenet_mean) / imagenet_std
        return image

    def predict(self, input_tensor):
        # Perform inference
        output_h5 = self.model.predict(input_tensor)
        output_list = output_h5.tolist()  # Convert NumPy array to Python list
        fix_list = output_list[0]
        fix_list = fix_list[:6]

        # Find the index of the highest value
        max_value = max(fix_list)
        max_index = output_list[0].index(max_value)

        # Map the index to the wound type
        wound_types = [
            "Stab_wound",
            "abrasion wound",
            "bruises wound",
            "burn wound",
            "cut wound",
            "laceration wound"
        ]
        return wound_types[max_index] if max_index < len(wound_types) else "No"
