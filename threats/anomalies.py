import numpy as np
from plot_funcs import plot_images


def anomaly(X, anomaly_type):
    num_samples = 40
    num_row = 4
    num_col = 10

    data = []
    
    for images in X:
        if anomaly_type == 'pixel_trap':
            indices = np.random.choice(images.shape[0], 3, replace=False)
            images[indices] = 0
            data.append(images)

        elif anomaly_type == 'row_add_logic':
            ind = int(images.shape[0]/2)-2
            images[ind+1] = images[ind]
            images[ind+2] = images[ind]
            images[ind+3] = images[ind]
            images[ind+4] = images[ind]
            data.append(images)
        
        elif anomaly_type == 'shifted_pixel':
            max_shift = 5
            m,n = images.shape[0], images.shape[1]
            col_start = np.random.randint(0, max_shift, images.shape[0])
            idx = np.mod(col_start[:,None] + np.arange(n), n)
            images = images[np.arange(m)[:,None], idx]
            data.append(images)
        
        else:
            print("anomaly type not found!!")
            break

    #data = np.array(data)
    #plot_images(data[:num_samples], [99]*num_samples, num_row=num_row, num_col=num_col)
    
    return np.array(data)