import requests

# Path to the image file you want to send
image_path = '../imgs/cellulitis.jpg'

# URL of the API endpoint where you want to send the image
url = 'http://127.0.0.1:5000/predict'

# Open the image file in binary mode
with open(image_path, 'rb') as image_file:
    # Define the files parameter for the POST request
    files = {'file': image_file}
    
    # Make the POST request to upload the image
    response = requests.post(url, files=files)
    
    # Print the response from the server
    print(response.status_code)
    print(response.text)