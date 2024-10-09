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


import requests

# Define the URL of the Flask endpoint
url = 'http://127.0.0.1:5000/predict'

files = {'file': open('C:\\gp-project\\imgs\\3rd.jpeg', 'rb')}

response = requests.post(url, files=files)

# Print the response from the server
print(response.status_code)  # Should be 200 if successful
print(response.json()) 