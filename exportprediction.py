from tensorflow.python.keras import models
from tensorflow.python.keras import losses
import tensorflow as tf
import functools
import os
import matplotlib.pyplot as plt
from skimage.measure import label as make_label
import numpy as np
import pandas as pd
from skimage.measure import regionprops
import cv2
from extras.cells_postprocessing import *
#%%

def show_window( event, img, lab, cells):
    
    
    # plot the images corresponding to the selected level
    #event.canvas.figure.axes[0].plot(x_data,y_data, 'b')
    
    ax = event.canvas.figure.axes[0]
        # ax.lines[0].remove() #Commented, otherwise I remove also the plot
    ax.cla()
    
    plot_back_img(ax,img)
    plot_circles(ax,cells)

    event.canvas.draw()


def find_closest( x0, y0, cells):
    
    
    d = (x0 - np.array(cells)[:,2])**2 + (y0 - np.array(cells)[:,1])**2
    
        
    return np.argmin(d)

def on_button_release(event,img,lab,cells):
    
    if event.button == 1:
        
        cells[-1][3] = np.sqrt( (cells[-1][1] - event.ydata)**2 +
                                (cells[-1][2] - event.xdata)**2)
        
    show_window( event, img, lab, cells)
        

def on_button_press( event, img,lab,cells, last_cell):
    
    
    # With a left click, add a cell    
    if event.button == 1:
        
        new_cell = [last_cell[0] + 1,
                    event.ydata,
                    event.xdata,
                    1]
    
        cells.append(new_cell)    
        last_cell[0] += 1
        
    # For a right click, remove the closest cell    
    if event.button == 3:
        # Decide which spot has to be removed
        idx = find_closest(event.xdata,event.ydata,cells)
        cells.pop(idx)# = np.delete(cells,idx,axis=0)        
    
    show_window( event, img, lab, cells)

def on_key_press( event, status, fnm):

    if event.key in worddict.keys():
        plt.close('all')
        
    status.append(event.key)  
    print('File {} marked this tile as {}.'.format(fnm,worddict[event.key]))
    
def plot_empty_vs_ml(img,lab,cells,status,fnm): 

    try:
        last_cell = [cells[-1][0]]
    except:
        last_cell = [-1]
        
#    batch_of_imgs = tf.keras.backend.get_session().run(next_element)[0]
#    predicted_labels = model.predict(batch_of_imgs)
#    labels = np.array([predicted_labels,predicted_labels])
#    img,lab= batch_of_imgs[0],labels[:,0,:,:,0]
    fig,ax = plt.subplots(ncols=2,figsize=(20, 10), sharex =True, sharey= True)

    for nplot in range(2):
        plot_back_img(ax[nplot],img)
        

    ax[1].contour(lab[:,:], levels = [0.5], colors = 'red', linestyles = ':')
    plot_circles(ax[1],np.array(cells))
#             ax[nplot].imshow(lab_img, cmap = 'hsv', alpha = 0.2)

    ax[1].set_title('ML Segmentation')
    
    # Why the lambda?? can I just put on_button_press? TODO
    fig.canvas.mpl_connect( 'button_press_event', lambda event: on_button_press( event, img,lab, cells , last_cell) )
    fig.canvas.mpl_connect( 'button_release_event', lambda event: on_button_release( event, img,lab, cells) )
    fig.canvas.mpl_connect( 'key_press_event', lambda event : on_key_press( event,status,fnm ) )

    fig.suptitle('Segmentation of {}'.format(fnm))
    plt.show()

    return cells

def plot_back_img(ax,img):

    ax.imshow(img[:,:,1]-img[:,:,0], vmin =0, vmax=10, cmap = 'Greys_r', origin = 'lower')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Corrected Segmentation')              

    
    return
        
def plot_circles(ax,cells):

    for ncell,cell in enumerate(cells):
    
        circle = plt.Circle((cell[2], cell[1]), cell[3], alpha = 0.5, edgecolor = 'yellow', fc = 'none', lw =2)
        ax.add_artist(circle)
        ax.text(cell[2] + cell[3], cell[1]+cell[3],int(cell[0]),horizontalalignment = 'left', verticalalignment = 'top', color ='yellow')
        
    return

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def get_baseline_dataset(filenames, 
                         preproc_fn=None,
                         threads=5, 
                         batch_size=1,
                         shuffle=True):           
  num_x = len(filenames)
  # Create a dataset from the filenames and labels
  dataset = tf.data.Dataset.from_tensor_slices((filenames,filenames))
  # Map our preprocessing function to every element in our dataset, taking
  # advantage of multithreading
  dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
  if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
    assert batch_size == 1, "Batching images must be of the same size"

  dataset = dataset.map(preproc_fn, num_parallel_calls=threads)
  
  if shuffle:
    dataset = dataset.shuffle(num_x)
  
  
  # It's necessary to repeat our data for all epochs 
  dataset = dataset.repeat().batch(batch_size)
  return dataset

def _process_pathnames(fname,fileid):
    # We map this function onto each pathname pair  
    img_str = tf.io.read_file(fname)
    img = tf.image.decode_png(img_str, channels = 3, dtype = tf.dtypes.uint16)
    
    return (img,fileid)

def _augment(img,fileid,
             resize=None,  # Resize the image to some size e.g. [256, 256]
             scale=1,  # Scale image e.g. 1 / 255.
             hue_delta=0,  # Adjust the hue of an RGB image by random factor
             horizontal_flip=False,  # Random left right flip,
             width_shift_range=0,  # Randomly translate the image horizontally
             height_shift_range=0):  # Randomly translate the image vertically 
  if resize is not None:
    # Resize both images
#    label_img = tf.image.resize_images(label_img, resize)
    img = tf.image.resize(img, resize)
  
  if hue_delta:
    img = tf.image.random_hue(img, hue_delta)
  
#  img, label_img = flip_img(horizontal_flip, img, label_img)
#  img, label_img = shift_img(img, label_img, width_shift_range, height_shift_range)
#  label_img = tf.to_float(label_img) * scale
  img = tf.cast(img, tf.float32) * scale 
  return (img,fileid)


def name_cells(label):
    
    lab_img = make_label(label > vmin)
    Lprop = regionprops(lab_img)
    cell_names = np.unique(lab_img)
    cell_names = cell_names[cell_names > 0] 
    cells = []
    cell_lab = 0           
    for ncell in cell_names:
        locs = np.where(lab_img == ncell)
        xctr,yctr = np.average(np.array(locs),axis = 1)
        xstd,ystd = np.std(np.array(locs),axis = 1)
        r = np.sqrt(xstd**2 + ystd**2)
        ecc = Lprop[ncell-1].eccentricity
        if r > rmin:
            cells.append([cell_lab,xctr,yctr,r,ecc])
            cell_lab +=1
    
    return cells


def decodename(name):
    
    return name.numpy().decode('utf-8').split('/')[-1]

def process_cells(c):
    
    if len(c) == 0:
        c.append([-1,np.nan,np.nan,np.nan,np.nan,np.nan])
    
    return c
    
#%%
if __name__ == '__main__':

    dataset = '2019_11_25'
    # dataset = '2019_02_20'
    model_path = '/data01/lorenzo/PROJECTS/cell_segmentation/models/train_wnewdata_{}epochs.hdf5'.format(99)
    rev_path = '/data01/lorenzo/PROJECTS/cell_segmentation/review/{}/'.format(dataset)

    # Alternatively, load the weights directly: model.load_weights(save_model_path)
    model = models.load_model(model_path, custom_objects={'bce_dice_loss': bce_dice_loss,
                                                               'dice_loss': dice_loss})
        
    worddict = {'p':'perfect',
                ' ':'perfect',
                'esape':'empty',
                'e':'empty',
                'b':'bad for training',
                'enter':'modified',
                'o':'other',
                'q':'quit'}
    #%%%
    
    user = 'ML'
    export_masks = False
    

    
    
    #%%
    # slices = np.arange(39,54)
    slices = [34,35,33] #np.arange(33,36)
    for N_slice in slices:
        img_dir = '/data01/lorenzo/PROJECTS/cell_segmentation/datasets/{}/tiled_slices/slice_{:04d}_tiled/'.format(dataset,N_slice)
        rev_path_sl = rev_path + 'MLpred/slice_{:04d}/'.format(N_slice)
        if not os.path.isdir(rev_path_sl):
            os.makedirs(rev_path_sl)

        rev_csv = rev_path + 'MLpred/slice_{:04d}.csv'.format(N_slice)

        try:
            df = pd.read_csv(rev_csv, sep = ';', decimal = ',')
            processed_filenames = df['filename'].values
            print('Reading processed images from {}...'.format(rev_csv))
        except:
            df = pd.DataFrame()
            processed_filenames = []

        #%%
        processed_filenames = []
        print(len(processed_filenames))
        print(len(np.unique(processed_filenames)))



        bunch_size = 99

        all_files = [f for f in os.listdir(img_dir) if f.endswith('overlap.png') and f not in processed_filenames]
        N_all_files = len(all_files)
        Nbunchs = int(N_all_files/bunch_size)

        for nb in range(Nbunchs):

            fnames = all_files[nb*bunch_size:(nb+1)*bunch_size]
            val_filenames = [os.path.join(img_dir, f) for f in fnames]
            batch_size = 3

            img_shape = [256,256]

            val_cfg = {
                'resize': [img_shape[0], img_shape[1]],
                'scale': 1 / 255. ,
            }
            val_preprocessing_fn = functools.partial(_augment, **val_cfg)

            N_files = len(val_filenames)
            print('Loading a dataset of {} files...'.format(N_files))
        #
            #%%
            val_ds = get_baseline_dataset(val_filenames,
                                           preproc_fn=val_preprocessing_fn,
                                           batch_size=batch_size,
                                           shuffle=False)

            rmin = 2
            vmin = 0.8

            pred_dir = img_dir + '_prediction'
            if not os.path.isdir(pred_dir):
                os.makedirs(pred_dir)

            # Running next element in our graph will produce a batch of images
            N_batchs = int(N_files/batch_size)
            stop = False
            first_file = fnames[0]
            for i,next_element in enumerate(val_ds):

                batch_of_imgs,batch_of_fnames = next_element
                predicted_labels = model.predict(batch_of_imgs)

                for l in range(batch_size):

                    status = []
                    this_file = decodename(batch_of_fnames[l])
                    if (this_file == first_file) & (i > 0):
                        status.append('q')
                        break

                    # initialize empty list for cells
                    cells = name_cells(predicted_labels[l,:,:,0])

                    if len(cells) > 0:
                        # add inout ratio
                        cells = compute_cell_contrast(batch_of_imgs[l], cells, outf=2)

        #                plot_empty_vs_ml(batch_of_imgs[l],predicted_labels[l,:,:,0],cells, status, this_file)
                        status.append('enter')
                        mask_fout = os.path.join(pred_dir,this_file.replace('.png','_mask.png'))

                        if status[0] == 'q':
                            break
                        #%
                    else:
                        status.append('e')
                        mask_fout = os.path.join(pred_dir,this_file.replace('.png','_mask_[empty].png'))

                    if export_masks:
                        lab_tile = predicted_labels[l,:,:,0] * 255.  # Essential to save the mask in binary way
                        cv2.imwrite(mask_fout, lab_tile)


                    this_df = (pd
                           .DataFrame(process_cells(cells), columns = ['index','x','y','r','eccentricity','contrast'])
                           .assign(filename=this_file)
                           .assign(slice_number=N_slice)
                           .assign(exit_status=status[0])
                           .assign(user=user)
                           .assign(date=pd.Timestamp.now())
                           )
                    this_df.to_csv(os.path.join(rev_path_sl,this_file.replace('png','csv')),
                                                sep = ';', decimal = ',', index = False)

                    df = df.append(this_df, ignore_index = True)

                    df.to_csv(rev_csv, sep = ';', decimal = ',', index = False)

                if status[0] == 'q':
                    break
            print('Bunch {} done, {} to go'.format(nb,Nbunchs-nb))
        print('Done!')

