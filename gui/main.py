import tkinter as tk
from os import path
from tkinter import filedialog, END


def browse_file():
     filepath = filedialog.askopenfilename()
     file = open(filepath, 'r')
     print(file.read())
     file.close()
     #my_label1.config(text="Selected file:")
     #print(filepath)
     #print("Selected file:", filepath)

def Instruction():
    my_label2.config(text="Instructions! Hey")

root = tk.Tk()
root.title("Motion Mapper")
root.geometry('800x500')

#Defined Image for Background
bg = tk.PhotoImage(file='bgblu.png')
my_labelBG = tk.Label(root, image=bg)
my_labelBG.place(x=0, y=0, relwidth=1, relheight=1)

start_button = tk.PhotoImage(file='StartB.png')
image_label = tk.Label()
image_label.pack(pady=20)

about_button = tk.PhotoImage(file='AboutB.png')
image_label = tk.Label()
image_label.pack(pady=20)

my_title= tk.Label(root,text="Motion Mapper", font=("Helvetica", 50), fg="black")
my_title.pack(pady=20)

#bg='#6b88fe' put after root in my_frame for the colour blue
my_frame = tk.Frame(root)
my_frame.pack(pady=20)

#Start Button
myStart_button = tk.Button(my_frame, image=start_button, command=browse_file)
myStart_button.grid(row=0, column=0, padx=20)
textarea = tk.Text(root)

my_label1 = tk.Label(root, text="")
my_label1.pack(pady=20)

#About Button
myAbout_button = tk.Button(my_frame, image=about_button, command=Instruction)
myAbout_button.grid(row=0, column= 1, padx=20)

my_label2 = tk.Label(root, text="")
my_label2.pack(pady=20)

tk.mainloop()
