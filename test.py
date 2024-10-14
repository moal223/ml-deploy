# from models.SkinDiseaseModel import SkinDisease
# from tensorflow.keras.preprocessing import image

# model = SkinDisease('./models/skin.h5')

# image_path = './3rd2.jpeg' 

# img = image.load_img(image_path, target_size=(224, 224))

# img = model.preprocessing(img)
# prediction = model.predict(img)
# print("Predicted class:", prediction)

# from fastai.learner import load_learner
# import pathlib
# from PIL import Image

# pathlib.PosixPath = pathlib.WindowsPath


# def load_image(image_file):
#     try:
#         img = Image.open(image_file)
#         img = img.convert("RGB")  # Ensure the image is in RGB format
#         return img
#     except Exception as e:
#         print(f"Error loading image: {e}")
#         return None

# model_path = './models/type.pkl'

# model = load_learner(model_path)

# image_path = r'C:\\gp-project\\imgs\\2rd.jpeg'
# image = load_image(image_path)

# if image is not None:
#     pred_class = model.predict(image)
#     print(pred_class[0], '\n')
# else:
#     print("Image could not be loaded. Please check the file path.")


# from models.SkinDiseaseModel import SkinDisease
# from PIL import Image

# skinModel = SkinDisease('models/skin.h5')


# def load_image(image_file):
#     try:
#         img = Image.open(image_file)
#         img = img.convert("RGB")  # Ensure the image is in RGB format
#         return img
#     except Exception as e:
#         print(f"Error loading image: {e}")
#         return None



# image_path = r'C:\\gp-project\\imgs\\chickenpox.jpeg'
# image = load_image(image_path)


# image_skin = skinModel.preprocessing(image)
# skin_response = skinModel.predict(image_skin)

# print(skin_response)


# from models.TypeModel import TypeModel
# from PIL import Image

# model = TypeModel('models/type.pkl')

# image_path = 'C:\\gp-project\\imgs\\3rd.jpeg'

# image = Image.open(image_path)
# # image = image.convert("RGB")

# result = model.predict(image)
# print(result)





# def preprocessing(image):
#     image_resize = image.resize((224, 224))  
#     image_array = np.array(image_resize)
#     image_array /= 255.0
#     return image_array

# image = Image.open('C:\\gp-project\\imgs\\chickenpox.jpeg')

# image_pre = preprocessing(image)



from tensorflow.keras.models import load_model
import numpy as np
# import cv2
from PIL import Image


model = load_model(r"./models/skin2.h5")




import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Read and prepare the new image for testing
def prepare_image(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (224, 224))
    img_scaled = img_resized / 255.0  # Scale the values to the range [0, 1]
    img_scaled = np.expand_dims(img_scaled, axis=0)  # Add a new dimension to make it (1, 224, 224, 3)
    return img_scaled

# Use the model to predict the new image
def predict_new_image(image_path):
    img = prepare_image(image_path)
    prediction = model.predict(img)
    
    # Extract the predicted class
    predicted_label = np.argmax(prediction) 
    confidence = np.max(prediction)
    
    # Print the result
    label_mapping = {0: 'cellulitis', 1: 'impetigo', 2: 'athlete-foot', 3: 'nail-fungus', 
                     4: 'ringworm', 5: 'cutaneous-larva-migrans', 6: 'chickenpox', 7: 'shingles'}
    print(f"The model predicts that this is: {label_mapping[predicted_label]} with confidence {confidence:.2f}")

image_path = r"C:\\gp-project\\imgs\\chickenpox.jpeg"
predict_new_image(image_path)
