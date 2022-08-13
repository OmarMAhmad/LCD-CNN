import numpy as np
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from keras.utils import load_img, img_to_array
from tensorflow import keras


class Window:
    """ Create A Control Window """
    def __init__(self, window):
        self.win = window
        self.win.geometry("1200x600+100+50")
        self.win.resizable(False, False)
        self.win.title("Lung Cancer Detection")

        # Background Image
        self.img = ImageTk.PhotoImage(Image.open(r"Images/BG_LCD.png"))
        bg_img = Label(self.win, image=self.img, bd=0)
        bg_img.place(x=0, y=0)

        # Button
        self.btn = Button(self.win, text="Get Started", command=self.control_screen, font=("Arial", 20, "bold"),
                          bg="#fff", fg="#02021b", activebackground="#02021b", activeforeground="#fff",
                          relief="flat", bd=0)
        self.btn.place(x=60, y=300, width=200, height=50)

        self.my_load_model = keras.models.load_model("Model/model.h5")
        # Button (Check Image)
        self.btn_img = Button(self.win, text="Check Image", command=self.image_check, font=("Arial", 15),
                              bg="#fff", fg="#02021b", activebackground="#02021b", activeforeground="#fff",
                              relief="flat", bd=0)

        # Label (Show Predict Result)
        self.lbl_pred = Label(self.win, bd=0, font=("Arial", 35, "bold"), bg="#02021b", relief="flat")

    def control_screen(self):
        self.btn.place_forget()
        self.btn_img.place(x=60, y=150, width=140, height=40)

    def image_check(self):
        img_path = filedialog.askopenfilename(initialdir="/", title="Select A Image",
                                              filetypes=(("Image png", "*.png*"), ("Image png", "*.png")))
        # lbl_path = ImageTk.PhotoImage(Image.open(img_path))

        # Making new Prediction ----------- [ Part Three ]
        test_image = load_img(img_path, target_size=(64, 64))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = self.my_load_model.predict(test_image)

        if result[0][0] == 1:
            self.lbl_pred.config(text="Normal", fg="#089b17")
        else:
            self.lbl_pred.config(text="Cancer", fg="#d33137")
        print(result)
        self.lbl_pred.place(x=100, y=300, width=300, height=100)


win = Tk()
obj_win = Window(win)
win.mainloop()
