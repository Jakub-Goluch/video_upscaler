import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from os import getcwd
from time import sleep

playback = False

def stop_video():
    playback = False
def start_video():
    playback = True

root = tk.Tk()
# window maximized
root.state('zoomed')

# creating frames
original_video = tk.Frame(root)
upscaled_video = tk.Frame(root)
controls = tk.Frame(root)
# position the frames
original_video.pack(side=tk.LEFT, expand=True)
upscaled_video.pack(side=tk.LEFT, expand=True)
controls.pack(side=tk.LEFT, expand=True)
canvas1 = tk.Canvas(original_video, width=(root.winfo_width() - 100)/2, height=root.winfo_height())
canvas1.pack()

# create the buttons
start_button = tk.Button(controls, text="START", command=start_video)
start_button.pack()
stop_button = tk.Button(controls, text="STOP", command=stop_video)
stop_button.pack()



root.mainloop()