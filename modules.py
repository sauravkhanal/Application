import os
import cv2
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from keras.models import load_model
import tensorflow as tf
from keras.callbacks import EarlyStopping 
import itertools
from PIL import Image

def extract_data_from_cheque(cheque_path :str)->None:
    folder_path = os.path.dirname(cheque_path)
    cheque = cv2.imread(cheque_path)
    ch_size = (1494,700)

    cheque = cv2.resize(cheque,ch_size)

    if os.path.isdir(folder_path) is False:
        os.makedirs(folder_path)

    sign1 = cheque[326:520, 705:1068]
    cv2.imwrite(f'{folder_path}/sign1.png',sign1)

    sign2 = cheque[326:520, 705+397:1068+397]
    cv2.imwrite(f'{folder_path}/sign2.png',sign2)

    amount = cheque[225:297, 1075:1468]
    cv2.imwrite(f'{folder_path}/amount.png',amount)

    name = cheque[145: 195, 310:310+1065]
    cv2.imwrite(f'{folder_path}/name.png',name)

    account_details = cheque[340:340+200, 95:95+370  ]
    cv2.imwrite(f'{folder_path}/account_details.png',account_details)

    amount_text = cheque[200:246, 156:156+905]
    cv2.imwrite(f'{folder_path}/amount_text.png',amount_text)

    #for date
    x,y = 45,71
    xo,yo = 1074,51
    margin = 7
    margin2 = 8
    margin3 = 15
    margin4 = 20
    d0 = cheque[yo:yo+y , xo:xo+x]
    d1 = cheque[yo:yo+y , xo+x*(2-1)+margin :xo+x*2 + margin]
    m0 = cheque[yo:yo+y , xo+x*(3-1)+margin2 :xo+x*3 + margin2]
    m1 = cheque[yo:yo+y , xo+x*(4-1)+margin2 :xo+x*4 + margin2]
    y0 = cheque[yo:yo+y , xo+x*(5-1)+margin3 :xo+x*5 + margin3]
    y1 = cheque[yo:yo+y , xo+x*(6-1)+margin3 :xo+x*6 + margin3]
    y2 = cheque[yo:yo+y , xo+x*(7-1)+margin4 :xo+x*7 + margin4]
    y3 = cheque[yo:yo+y , xo+x*(8-1)+margin4 :xo+x*8 + margin4]

    cv2.imwrite(f'{folder_path}/d0.png',d0)
    cv2.imwrite(f'{folder_path}/d1.png',d1)
    cv2.imwrite(f'{folder_path}/m0.png',m0)
    cv2.imwrite(f'{folder_path}/m1.png',m1)
    cv2.imwrite(f"{folder_path}/y0.png",y0)
    cv2.imwrite(f"{folder_path}/y1.png",y1)
    cv2.imwrite(f"{folder_path}/y2.png",y2)
    cv2.imwrite(f"{folder_path}/y3.png",y3)

def extract_signature_from_image(image_path : str ,size:tuple= (300,300),margin:int = 15):
    image_name = os.path.basename(image_path)
    image_name = image_name.split('.')[0]
    save_path = os.path.join(os.path.split(image_path)[0], f"extracted_from_{image_name}")
    if os.path.isdir(save_path) is False: #split at . to remove .png extension from folder name
        os.makedirs(save_path)
        #dir_made = True
    
    signature_grid = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    signature_grid = cv2.resize(signature_grid,(size[0]*5,size[0]*7))

    y_adj =0
    x_adj = 0
    rows = 7
    cols = 5
    xo,yo = (0,0)#top left corner ie: starting point
    x,y = (0,0)
    width = size[0] #width , x
    height = size[1] #height , y
    desired_size = size[0]#<----------------------------------------
    index = 1 #for incrementing name of image
    # dir_made = False
    # adj = 0 # in order to adjust

    for i in range(rows):
        for j in range(cols):

            cropped_image = signature_grid[y + margin: y - margin + height, x + margin: x - margin + width]

            # crop rectangle part only
            #gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            gray = cropped_image.copy()
            # To invert the text to white
            gray = 255*(gray < 128).astype(np.uint8)
            coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
            # Find minimum spanning bounding box
            xa, ya, wa, ha = cv2.boundingRect(coords)
            # Crop the image - note we do this on the original image
            rect = cropped_image[ya:ya+ha, xa:xa+wa]

            # now add paddinga and make square
            var = rect.shape
            old_size = var[:2]
            ratio = float(desired_size)/max(old_size)
            new_size = tuple([int(xa*ratio) for xa in old_size])
            im = cv2.resize(rect, (new_size[1], new_size[0]))
            delta_w = desired_size - new_size[1]
            delta_h = desired_size - new_size[0]
            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)
            color = [255, 255, 255]
            new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

            cv2.imwrite(f'{save_path}/sign{index}.png',new_im)
            x = x + width #next gird in same row
            index+=1 #increment for namea

        x = xo #reset x to the leftmost border

        y = y + height #increment y to next row
    preprocess_single_folder(save_path,save_path)

def preprocess_single_folder(folder_path,save_folder_path, final_img_size = (200,200), power_law=False, segment=True, log_transform=False):
  image_batch = os.listdir(folder_path)
  image_data = [x for x in image_batch if x.endswith('png') or x.endswith('PNG') or x.endswith('jpg') or x.endswith('JPG')]

  for sample in tqdm(image_data):
    img_path = os.path.join(folder_path, sample)
    #importing images from drive
    #x = image.load_img(img_path)
    #img = image.img_to_array(x)
    img = cv2.imread(img_path)
        
    #Perfom Median blur on image
    mbvalue = int(np.max(img.shape)/200)
    mbvalue = mbvalue if mbvalue%2==1 else mbvalue+1
    img = cv2.medianBlur(img, mbvalue)

    #changing RGB to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    #resize image to 600xH
    img = cv2.resize(img, (600, int(600*float(img.shape[0])/img.shape[1])))
    
    #if power_law enabled
    if(power_law):
      img = img**0.9
      img[img>255]=255
      img[img<0]=0
      img = img.astype('uint8')
          
    #denoising the grayscale image
    img = cv2.fastNlMeansDenoising(img, None, 10, 21)
    
    if (log_transform):
        img = (np.log(img+1)/(np.log(10+np.max(img))))*255
        img=img.astype('uint8')
    
    #Threshold binary image
    avg = np.average(img)
    _,image = cv2.threshold(img, int(avg)-30, 255, cv2.THRESH_BINARY)
            
    #segment the signature section only
    if(segment):
      seg = segmentImage(image)
      image = image[seg[2]:seg[3], seg[0]:seg[1]]
          
    #padding to make image into square
    lp, rp, tp, bp = (0,0,0,0)
    if(image.shape[0]>image.shape[1]):
      lp = int((image.shape[0]-image.shape[1])/2)
      rp = lp
    elif(image.shape[1]>image.shape[0]):
      tp = int((image.shape[1]-image.shape[0])/2)
      bp = tp
    image_padded = cv2.copyMakeBorder(image, tp, bp, lp, rp, cv2.BORDER_CONSTANT, value=255)

    #resizing the image
    img = cv2.resize(image_padded, final_img_size)

    #producing image negative
    img = 255-img

    #skeletonizing image
    #img = thin(img/255)

    #img = img.astype('bool')
    # cv2.imshow(sample,img)
    # cv2.waitKey(500)
    new_save_path = os.path.join(save_folder_path , sample)
    if not os.path.isdir(save_folder_path):
        os.makedirs(save_folder_path) 
    #print(str(new_save_path))
    cv2.imwrite(new_save_path, img)

def segmentImage(image):  
  hHist=np.zeros(shape=image.shape[0], dtype=int)
  vHist=np.zeros(shape=image.shape[1], dtype=int)

  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      if(image[i][j]==0):
        hHist[i]+=1
        vHist[j]+=1
  
  locLeft=0
  locRight=image.shape[0]
  locTop=0
  locBottom=image.shape[1]
  
  count=0
  for i in range(hHist.shape[0]):
    if(count<=0):
        count=0
        if(hHist[i]!=0):
            locTop=i
            count+=1
    else:
        if(hHist[i]!=0):
            count+=1
        else:
            count-=hHist.shape[0]/100

        if(count>hHist.shape[0]/30):
            break
            
  count=0
  for i in reversed(range(hHist.shape[0])):
    if(count<=0):
        count=0
        if(hHist[i]!=0):
            locBottom=i
            count+=1
    else:
        if(hHist[i]!=0):
            count+=1
        else:
            count-=hHist.shape[0]/100

        if(count>hHist.shape[0]/30):
            break
            
  count=0
  for i in range(vHist.shape[0]):
    if(count<=0):
        count=0
        if(vHist[i]!=0):
            locLeft=i
            count+=1
    else:
        if(vHist[i]!=0):
            count+=1
        else:
            count-=vHist.shape[0]/100

        if(count>vHist.shape[0]/30):
            break
            
  count=0
  for i in reversed(range(vHist.shape[0])):
    if(count<=0):
        count=0
        if(vHist[i]!=0):
            locRight=i
            count+=1
    else:
        if(vHist[i]!=0):
            count+=1
        else:
            count-=vHist.shape[0]/100

        if(count>vHist.shape[0]/30):
            break
            
  return locLeft, locRight, locTop, locBottom

def make_train_dataset (image_dir1:str, image_dir2:str)->list:
    '''
    path must have / at end 
    to be sorted at final
    '''
    dataset = []
    label = []
    #image_dir = input("PROMPT TO select image directory")
    images = os.listdir(image_dir1)
    for i,image_name in enumerate(images):
        if (image_name.split('.')[1] == 'png'):
            image = cv2.imread(os.path.join(image_dir1,image_name), cv2.COLOR_BGR2GRAY)
            image = Image.fromarray(image)
            if(image.height != image_size[0] and image.width != image_size[1]):
                image = image.resize((SIZE, SIZE))
            dataset.append(np.array(image))
            label.append(0)
    
    images = os.listdir(image_dir2)
    for i,image_name in enumerate(images):
        if (image_name.split('.')[1] == 'png'):
            image = cv2.imread(os.path.join(image_dir2,image_name), cv2.COLOR_BGR2GRAY)
            image = Image.fromarray(image)
            image = image.resize((SIZE, SIZE))
            dataset.append(np.array(image))
            label.append(1)

    return dataset,label #needs to be converted into numpy array

def make_paired_dataset(X, y):
  X_pairs, y_pairs = [], []

  tuples = [(x1, y1) for x1, y1 in zip(X, y)]
  
  for t in itertools.product(tuples, tuples):
    pair_A, pair_B = t
    img_A, label_A = t[0]
    img_B, label_B = t[1]

    new_label = int(label_A == label_B)

    X_pairs.append([img_A, img_B])
    y_pairs.append(new_label)
  
  X_pairs = np.array(X_pairs)
  y_pairs = np.array(y_pairs)

  return X_pairs, y_pairs

SIZE = 50
image_size = (SIZE,SIZE)

def train_model(image_folder_1:str, image_folder_2:str ,num_epochs :int = 3, untrained_model_path:str = './data/untrained_model.h5'):
    '''
    returns trained model for given dataset
    donot forget to save the model
    '''
    dataset,label = make_train_dataset(image_folder_1, image_folder_2)
    
    dataset = np.array(dataset)
    label  =np.array(label)

    model = load_model(untrained_model_path) # untrained model path must be given here

    model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

    data_pairs, label_pairs = make_paired_dataset(dataset,label)

    model.fit(x=[ data_pairs[:, 1, :, :],data_pairs[:, 0, :, :]],
          y=label_pairs,
          validation_split = 0.2,
          epochs=num_epochs,
          batch_size=32,
          callbacks=[EarlyStopping(patience=2)],
         )

    return model

def make_test_dataset(file_path1:str, file_path2:str)->list:
    '''
    takes two image to calculate similarity
    '''

    dataset = []

    image = cv2.imread(file_path1,cv2.COLOR_BGR2GRAY)
    image = Image.fromarray(image)
    image = image.resize((SIZE,SIZE))
    dataset.append(np.array(image))

    image = cv2.imread(file_path2,cv2.COLOR_BGR2GRAY)
    image = Image.fromarray(image)
    image = image.resize((SIZE,SIZE))
    dataset.append(np.array(image))

    return dataset

def predict(model_path,image_A:str, image_B:str)->int:#returns similarity in %
    
    data  = make_test_dataset(image_A, image_B)
    data = np.array(data)

    model = load_model(model_path)

    prediction = model.predict([data[0].reshape((1, SIZE,SIZE)), 
               data[1].reshape((1, SIZE,SIZE))])

    return prediction[0]*100
