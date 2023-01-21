from modules import *
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox

ctk.set_appearance_mode('system')
ctk.set_default_color_theme('dark-blue')
ctk.set_widget_scaling(1.5)

root = ctk.CTk()
root.geometry = ('500x700')
root.title = ('Signature Verificaton System')

def button0_command():#extract  300*300
    try:
        filename = ctk.filedialog.askopenfilename()
        extract_signature_from_image(filename, (300,300),15, folder_given = True,save_folder = './data/extracted_signatures')
        messagebox.showinfo('','Extraction Complete')
    except Exception as e:
        messagebox.showwarning('Error occured',f'{e}')

def button1_command():# train model
    
    image_dir_0 = ctk.filedialog.askdirectory(
                                           )
    
    image_dir_1 = './data/signature/train/99'

    if os.path.isdir(image_dir_1) is False:
        directory_not_found_popup(image_dir_1)

        image_dir_1 = ctk.filedialog.askdirectory(root,
                                               initialdir='./',
                                               title = 'Please select a directory of signature class 2')
        
    messagebox.showinfo('',"Model is training\nEstimated time : 150 seconds")
    model = train_model(image_dir_0, image_dir_1)

    check_and_create_dir('.data/trained_model')

    model_name = f'{os.path.basename(image_dir_0)}vs{os.path.basename(image_dir_1)}'

    model.save(f'./data/trained_model/{model_name}.h5')

    messagebox.showinfo('Training completes',f'Model saved as: .data/trained_model/{model_name}.h5')

def button2_command():# extract cheque
    filename = ctk.filedialog.askopenfilename()
    extract_data_from_cheque(filename,save_path='./data', folder_given = True)
    messagebox.showinfo('','Extraction completed')


def button3_command():
    pass



button0 = ctk.CTkButton(root,
                        text = 'Extract Signature',
                        command = button0_command
                        )

button1 = ctk.CTkButton(root,
                        text = 'Train Model',
                        command = button1_command
                        )

button2 = ctk.CTkButton(root,
                        text = 'Scan Cheque',
                        command = button2_command
                        )

button3 = ctk.CTkButton(root,
                        text = 'Test signature',
                        command = button3_command
                        )

button4 = ctk.CTkButton(root,
                        text = 'Quit',
                        command = root.quit
                        )

button0.pack(padx=0, pady=5)
button1.pack(padx=0, pady=5)
button2.pack(padx=0, pady=5)
button3.pack(padx=0, pady=5)
button4.pack(padx=0, pady=5)

def directory_not_found_popup(directory_path:str):
    messagebox.showwarning('Path error', f'{directory_path} not found')

def check_and_create_dir(dir_path:str):
    if os.path.isdir(dir_path) is False:
        os.makedirs(dir_path)


root.mainloop()
