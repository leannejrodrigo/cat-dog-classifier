from PIL import Image
import os

folders = [
    "dataset/cats",
    "dataset/dogs"
]

for folder in folders:
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)

        try:
            img = Image.open(path)
            img = img.convert("RGB")
            img.save(path)

        except Exception as e:
            print("Deleting bad file:", path)
            os.remove(path)

print("Dataset cleaned successfully.")

