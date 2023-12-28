from openclip import OpenCLIPEmbeddings
import base64
import time

openclip = OpenCLIPEmbeddings()
# Embedding text
texts = ["A photo of a cat."]
start_time_text = time.time()
text_features = openclip.embed_texts(texts)
end_time_text = time.time()
print("Text embedding time:", end_time_text - start_time_text, "seconds")
# print(text_features)

image = "images/cat.jpeg"
start_time_image = time.time()
image_features = openclip.embed_images([image])
end_time_image = time.time()
print("Image embedding time:", end_time_image - start_time_image, "seconds")
# print(image_features)

with open(image, "rb") as f:
    base64_image = base64.b64encode(f.read()).decode("utf-8")
start_time_base64 = time.time()
base64_features = openclip.embed_base64s([base64_image])
end_time_base64 = time.time()
print("Base64 embedding time:", end_time_base64 - start_time_base64, "seconds")
# print(base64_features)

print(image_features == base64_features)