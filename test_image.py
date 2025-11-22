from backend.image_emotion import analyze_image
from PIL import Image

img = Image.open("test_face.jpg")
print(analyze_image(img))
