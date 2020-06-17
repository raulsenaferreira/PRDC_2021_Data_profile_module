import numpy as np


def anomaly(images, anomaly_type):
    if anomaly_type == 'pixel_trap':
        indices = np.random.choice(images.shape[0], 2, replace=False)
        images[indices] = 0

    elif anomaly_type == 'row_add_logic':
        ind = int(images.shape[0]/2)-2
        images[ind+1] = images[ind]
        images[ind+2] = images[ind]
        images[ind+3] = images[ind]
        images[ind+4] = images[ind]
    
    elif anomaly_type == 'shifted_pixel':
        max_shift = 5
        m,n = images.shape[0], images.shape[1]
        col_start = np.random.randint(0, max_shift, images.shape[0])
        idx = np.mod(col_start[:,None] + np.arange(n), n)
        images = images[np.arange(m)[:,None], idx]

    return images