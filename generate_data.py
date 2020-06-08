import util


def drift_data(train, test, drift_type, mode='load', persist_data = False):
        success = persist_data
        (x_train, y_train) = train
        (x_test, y_test) = test

        # input image dimensions for doing some types of drift
        img_rows, img_cols, img_dim = 64, 64, 1
        if mode=='generate':
            if drift_type == 'cvt' or drift_type == 'cht' or drift_type == 'cdt':
                x_train, y_train = util.generate_data_translations(x_train, y_train, drift_type)
                x_test, y_test = util.generate_data_translations(x_test, y_test, drift_type)
            
            elif drift_type=='rotated':
                x_train, y_train = util.rotating_data(x_train, y_train)
                x_test, y_test = util.rotating_data(x_test, y_test)
                img_rows, img_cols, img_dim = 28, 28, 1
            
            elif drift_type=='back_round':
                (x_train, y_train), (x_test, y_test) = util.load_mnist_rand_back()

            elif drift_type=='moving':    
                (x_train, y_train), (x_test, y_test) = util.load_batches_moving_mnist()  

            #elif drift_type=='extended':
                #x_train, x_test = util.reshaping_data(x_train, x_test, img_rows, img_cols, img_dim)
                #correcting a bug with rotation of the images in this dataset
                #x_test, y_test = util.rotating_data(x_test, y_test, 270)
                #x_test = mirroring_image(x_test)

            #elif drift_type=='affnist':    
                #(x_train, y_train), (x_test, y_test) = util.load_batches_affnist()
                #x_train, x_test = util.reshaping_data(x_train, x_test, 40, 40, img_dim)

            #reshaping data
            x_train, x_test = util.reshaping_data(x_train, x_test, img_rows, img_cols, img_dim) 
        
            if persist_data:
                success = util.save_data(x_train, y_train, x_test, y_test, drift_type)
                
                return success
        