from modules import *
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox


'''
remaining task:
make all features computer independent ,
first check if .data/needed_file exists 
if no, popup that file doesn't exist and open select file
after operation is complete show file location and open it 
(can add radio button to show file locations or not automatically)

It may be difficult to add console 
if difficult make exe with a console

Theory is most important 

MUST UNDERSTAND ALL THE CONCEPTS CLEARLY
WHY THIS MODEL 
WHY THIS ARHITECTURE 
WHAT DOES THIS LOSS FUNCTION DO?


'''

ctk.set_appearance_mode('system')
ctk.set_default_color_theme('dark-blue')
ctk.set_widget_scaling(1.5)

root = ctk.CTk()
root.geometry = ('500x700')
root.title = ('Signature Verificaton System')

def button0_command():#extract  300*300
    try:
        filename = ctk.filedialog.askopenfilename()

        save_in_custom_folder = messagebox.askyesno('','Do you want to save in custom folder?')

        if save_in_custom_folder is False:
            image_name = os.path.basename(filename)
            image_name = image_name.split('.')[0]
            save_path = f'./data/extracted_signatures/extracted_from_{image_name}'
            save_path =  os.path.abspath(save_path)

            try:
                if os.path.isdir(save_path) is False:
                    os.makedirs(save_path)
            except Exception as e:
                print(f"Couldn't make directory: {save_path}\n error : {e}")
                
        else:
            save_path = ctk.filedialog.askdirectory()

        return_path = extract_signature_from_image(filename, size=(300,300),margin = 15,
                                                 save_path=save_path)
        
        messagebox.showinfo('Extraction Complete', f'Image saved at : {return_path}')
        os.startfile(return_path)

    except Exception as e:
         messagebox.showwarning('Error occured',f'{e}')

def button1_command():# train model
    
    image_dir_0 = ctk.filedialog.askdirectory()
    
    image_dir_1 = './data/signature/train/99'

    if os.path.isdir(image_dir_1) is False:
        directory_not_found_popup(image_dir_1)

        image_dir_1 = ctk.filedialog.askdirectory(root,
                                               initialdir='./',
                                               title = 'Please select a directory of signature class 2')
        
    messagebox.showinfo('',"Model is training\nEstimated time : 150 seconds")
    model = train_model(image_dir_0, image_dir_1,num_epochs=5)

    check_and_create_dir('.data/trained_model')

    model_name = f'{os.path.basename(image_dir_0)}vs{os.path.basename(image_dir_1)}'

    model.save(f'./data/trained_model/{model_name}.h5')

    messagebox.showinfo('Training completes',f'Model saved as: .data/trained_model/{model_name}.h5')

def button2_command():# extract cheque
    filename = ctk.filedialog.askopenfilename()
    extract_data_from_cheque(filename,save_path='./data', folder_given = True)
    messagebox.showinfo('','Extraction completed')


def test_model_selection(trained_model_name):

    button3.grid_remove()

    ctk.CTkLabel(root,text = 'Select user to test').grid(row = 4, column = 1)

    trained_model_path = os.path.join('./data/trained_model',trained_model_name)

    reference_image = os.path.join('./data/signature/train',trained_model_name.split('vs')[0],'sign1.png')

    if os.path.isfile(reference_image) is False:
        messagebox.showwarning('file not found', 'path: ' + reference_image)

    image_to_test = ctk.filedialog.askopenfilename()

    output = predict(trained_model_path,reference_image,image_to_test)
    messagebox.showinfo('Accuracy',f'The similarity of signature is: {output}')



def button3_command(): # test signature

    file_list = os.listdir('./data/trained_model')
    option_box = ctk.CTkOptionMenu(root,
                                values = file_list,
                                button_color='green',
                                button_hover_color='blue',
                                fg_color='grey' ,
                                command= test_model_selection
                                )
    option_box.grid(row = 3 , column = 1, pady = 10)



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



button0.grid(padx=0, pady=5, row = 0 , column = 1 )
button1.grid(padx=0, pady=5, row = 1 , column = 1 )
button2.grid(padx=0, pady=5, row = 2 , column = 1 )
button3.grid(padx=0, pady=5, row = 4 , column = 1 )
button4.grid(padx=0, pady=5, row = 5 , column = 1 )



# ctk.CTkOptionMenu(master=root, values= lis).grid(row = 4, column = 2)

# lb = tk.Listbox(root)
# for i, items in enumerate(lis):
#     lb.insert(i,items)
# lb.grid(row=4, column=0 )

def directory_not_found_popup(directory_path:str):
    messagebox.showwarning('Path error', f'{directory_path} not found')

def check_and_create_dir(dir_path:str):
    if os.path.isdir(dir_path) is False:
        os.makedirs(dir_path)


root.mainloop()
