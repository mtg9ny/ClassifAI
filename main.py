import os
from tkinter import Tk, Button, Label, filedialog
from PIL import Image, ImageTk
from keras.preprocessing import image
from keras.models import load_model
from keras.optimizers import Adam

# load model
model = load_model('trained_model.h5')

# button click function
def classify_button_click():
    # ask for img file
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # display image in gui
        display_image(file_path)
        # classify img
        classification = classify_image(file_path)
        # display result in gui
        result_label.configure(text=f"This image is a {classification}!", font=("Arial", 16))

# display img function
def display_image(image_path):
    img = Image.open(image_path)
    img = img.resize((300, 300))
    photo = ImageTk.PhotoImage(img)
    image_label.configure(image=photo)
    image_label.image = photo

# classify img function
def classify_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = img.reshape(1, 224, 224, 3)
    prediction = model.predict(img)
    class_label = "cat" if prediction[0][0] < 0.5 else "dog"
    return class_label

# create gui
window = Tk()
window.title("Image Classifier")
window.geometry("500x400")

# button for img upload
classify_button = Button(window, text="Select Image", command=classify_button_click, font=("Arial", 14))
classify_button.place(relx=0.5, rely=0.1, anchor="center")

# label img
image_label = Label(window)
image_label.place(relx=0.5, rely=0.4, anchor="center")

# label for res
result_label = Label(window, text="", font=("Arial", 16))
result_label.place(relx=0.5, rely=0.8, anchor="center")

# open gui
window.mainloop()