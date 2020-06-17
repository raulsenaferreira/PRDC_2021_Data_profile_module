import sys
import numpy as np
from scipy import ndimage
from PIL import Image
from skimage.transform import resize


def get_separator():
    is_windows = sys.platform.startswith('win')
    sep = '\\'

    if is_windows == False:
        sep = '/'

    return sep

#globals
sep = get_separator()
img_rows = 28
img_cols = 28
img_dim = 1


def rotating_data(images, labels, correcting=None):

    X = []
    y = []
    bg_value = -0.5 # this is regarded as background's value black
    
    for image, label in zip(images, labels):

        if correcting==None:

            # register original data
            X.append(image)
            y.append(label)
            
            angles = [-45, -22.5, 22.5, 45]
            
            for angle in angles:
                
                new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)
                # register new training data
                #new_img = img.resize((28, 28), Image.ANTIALIAS)
                #new_img = np.array(new_img)
                X.append(new_img)
                y.append(label)
        else:
            new_img = ndimage.rotate(image,correcting,reshape=False, cval=bg_value)
            # register new training data
            X.append(new_img)
            y.append(label)            

    # return them as arrays
    return np.asarray(X), np.asarray(y)


def transformations(name):
        
    step_size = 4 
    boundary = 20 #image limits from the center
    repetitions = int(boundary/2)+1 #number of interpolations

    a = [1]*repetitions
    b = [0]*repetitions
    c = [0]*repetitions #left/right (i.e. 5/-5)
    d = [0]*repetitions
    e = [1]*repetitions
    f = [0]*repetitions #up/down (i.e. 5/-5)

    if name == 'cht':
        c = np.arange(boundary, -boundary-step_size, -step_size).tolist()
    elif name == 'cvt':
        f = np.arange(-boundary, boundary+step_size, step_size).tolist()
    elif name == 'cdt':
        c = np.arange(boundary, -boundary-step_size, -step_size).tolist()
        f = np.arange(-boundary, boundary+step_size, step_size).tolist()

    return (a,b,c,d,e,f)


def generate_data_translations(images, labels, name, row, col, dim):

    X = []
    y = []

    pixels_added = 18
    transf = transformations(name)
    num_interp = len(transf[0])

    for image, label in zip(images, labels):
        #increasing image
        if dim == 1:
            image = np.pad(image, ((pixels_added,pixels_added),(pixels_added,pixels_added)), 'constant')
        elif dim == 3:
            image = np.pad(image, ((pixels_added,pixels_added),(pixels_added,pixels_added), (0, 0)), 'constant')    
        
        for n in range(0, num_interp):
            img = None
            #print(transf[0][n])
            t = (transf[0][n], transf[1][n], transf[2][n], transf[3][n], transf[4][n], transf[5][n])
            #print(t)
            if dim == 1:
                img = Image.fromarray(image)
            elif dim == 3:
                img = Image.fromarray(image.astype('uint8'), 'RGB')

            new_img = img.transform(img.size, Image.AFFINE, t)
            #new_img = resize(new_img, (32, 32))
            new_img = img.resize((row, col), Image.ANTIALIAS)
            new_img = np.array(new_img)
            # register new training data
            X.append(new_img)
            y.append(label)

    # return them as arrays
    return np.asarray(X), np.asarray(y)