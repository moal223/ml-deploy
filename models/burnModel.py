import tensorflow as tf

class BurnModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict_burn_degree(self, input_tensor):
        # Perform the prediction
        output_h5 = self.model.predict(input_tensor)
        output_list = output_h5.tolist()  # Convert NumPy array to Python list

        # Determine the burn degree based on the prediction
        max_value = max(output_list[0])
        max_index = output_list[0].index(max_value)

        if max_value < 0:
            return "No"
        elif max_index == 0:
            return "first degree"
        elif max_index == 1:
            return "second degree"
        elif max_index == 2:
            return "third degree"

        return "No"