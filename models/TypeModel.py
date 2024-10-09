from fastai.learner import load_learner
import pathlib

# Update pathlib to use WindowsPath
pathlib.PosixPath = pathlib.WindowsPath

class TypeModel:
    def __init__(self, model_path):
        """
        Initialize the ImageClassifier with the model path.
        
        Args:
            model_path (str): Path to the trained model.
        """
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        """Load the model from the specified path."""
        try:
            model = load_learner(self.model_path)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None


    def predict(self, image):
        """
        Make a prediction on the provided image file path.
        
        Args:
            image_path (str): Path to the image file.
        
        Returns:
            str: Predicted class or an error message.
        """
        if image is not None:
            image_rgb = image = image.convert("RGB")
            pred_class = self.model.predict(image_rgb)
            return pred_class[0]
        else:
            return "Image could not be loaded. Please check the file path."