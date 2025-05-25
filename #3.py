# -*- coding: utf-8 -*-
"""
Created on Fri May 17 19:54:38 2024

@author: marina
"""


import tkinter as tk #used for creating gui
from tkinter import filedialog # provide dialogs to open files
from tkinter import ttk # TK themed widgets --> used for sliders
import cv2 
import numpy as np # library for numerical operations on arrays
from PIL import Image, ImageTk # used to convert images to a format tkinter can display

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")

        self.original_image = None
        self.display_image = None 

        self.create_widgets()

    def create_widgets(self):
        # Frame for buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Load image button
        load_button = tk.Button(button_frame, text="Load Image", command=self.load_image)
        load_button.pack(side=tk.TOP, padx=5,  pady=5)
        
        open_button = tk.Button(button_frame, text="Open", command=self.apply_open)
        open_button.pack(side=tk.TOP, padx=5, pady=5)

        close_button = tk.Button(button_frame, text="Close", command=self.apply_close)
        close_button.pack(side=tk.TOP, padx=5, pady=5)

        hough_button = tk.Button(button_frame, text="Hough Circles", command=self.apply_hough_circle_transform)
        hough_button.pack(side=tk.TOP, padx=5, pady=5)

        thresholding_button = tk.Button(button_frame, text="Thresholding", command=self.apply_thresholding_segmentation)
        thresholding_button.pack(side=tk.TOP, padx=5, pady=5)

        # Slider frame
        self.slider_frame = tk.Frame(self.root)
        self.slider_frame.pack(side=tk.TOP, fill=tk.X)

        # Image display
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.display_image = self.original_image.copy()
            self.update_image(self.display_image)

    def update_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.image_label.config(image=image)
        self.image_label.image = image

    def add_slider(self, label, from_, to, initial, command):
        """
        Adds a slider to the GUI for selecting filter parameters.
        
        Args:
            label: Text label for the slider.
            from_: Minimum value of the slider.
            to: Maximum value of the slider.
            initial: Initial value of the slider.
            command: Command to be executed when the slider value changes.
        """
        for widget in self.slider_frame.winfo_children():
            widget.destroy()
        slider_label = tk.Label(self.slider_frame, text=label)
        slider_label.pack(side=tk.LEFT, padx=5)
        slider = tk.Scale(self.slider_frame, from_=from_, to=to, orient=tk.HORIZONTAL, command=command)
        slider.set(initial)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
    
    def apply_open(self):
        """
        Adds a slider for the kernel size and applies the opening morphological operation to the image.
        """
        self.add_slider("Kernel Size", 1, 20, 5, self.update_open)

    def update_open(self, value):
        """
        Updates the image with the opening morphological operation applied.
        
        Args:
            value: Kernel size for the opening operation.
        """
        kernel_size = int(value)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        open_image = cv2.morphologyEx(self.original_image, cv2.MORPH_OPEN, kernel)
        self.update_image(open_image)

    def apply_close(self):
        """
        Adds a slider for the kernel size and applies the closing morphological operation to the image.
        """
        self.add_slider("Kernel Size", 1, 20, 5, self.update_close)

    def update_close(self, value):
        """
        Updates the image with the closing morphological operation applied.
        
        Args:
            value: Kernel size for the closing operation.
        """
        kernel_size = int(value)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        close_image = cv2.morphologyEx(self.original_image, cv2.MORPH_CLOSE, kernel)
        self.update_image(close_image)

    def apply_hough_circle_transform(self):
        """
        Applies the Hough Circle Transform to detect circles in the image.
        """
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            hough_image = self.original_image.copy()
            for i in circles[0, :]:
                cv2.circle(hough_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(hough_image, (i[0], i[1]), 2, (0, 0, 255), 3)
            self.update_image(hough_image)
            
            
    def apply_thresholding_segmentation(self):
        """
        Applies thresholding-based segmentation to the image.
        """
        # Uncomment the following lines if using a slider for threshold
        # self.slider_frame.pack(side=tk.TOP, fill=tk.X)
        # self.add_slider("Threshold", 0, 255, 127, self.update_thresholding_segmentation)

        threshold_value = 127  # Or get value from slider if used
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        _, threshold_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
        self.update_image(cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2BGR))       

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()
