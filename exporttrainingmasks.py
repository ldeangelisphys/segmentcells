from tensorflow.python.keras import models
from tensorflow.python.keras import losses
import tensorflow as tf
import functools
import os
import matplotlib.pyplot as plt
from skimage.measure import label as make_label
import numpy as np
import pandas as pd
import cv2
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
    
def plot_empty_vs_ml(img,oldlab,newlab,cells,cleancells,fnm): 

    try:
        last_cell = [cells[-1][0]]
    except:
        last_cell = [-1]
        
    fig,ax = plt.subplots(ncols=2,figsize=(20, 10), sharex =True, sharey= True)

    for nplot in range(2):
        plot_back_img(ax[nplot],img)
        
    for nj,tlab in enumerate([oldlab,newlab]):
        ax[nj].contour(tlab[:,:], levels = [0.5], colors = 'red', linestyles = ':')
    for nj,tcells in enumerate([cells,cleancells]):
        plot_circles(ax[nj],np.array(tcells))
    #             ax[nplot].imshow(lab_img, cmap = 'hsv', alpha = 0.2)

    ax[1].set_title('ML Segmentation')
    
    fig.suptitle('Segmentation of {}'.format(fnm))
    plt.savefig(rev_path_ov + fnm)
    plt.close('all')

    return

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
    img_str = tf.read_file(fname)
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
    img = tf.image.resize_images(img, resize)
  
  if hue_delta:
    img = tf.image.random_hue(img, hue_delta)
  
#  img, label_img = flip_img(horizontal_flip, img, label_img)
#  img, label_img = shift_img(img, label_img, width_shift_range, height_shift_range)
#  label_img = tf.to_float(label_img) * scale
  img = tf.cast(img, tf.float32) * scale 
  return (img,fileid)


def name_cells(label, deleted_labs = [], cleanlabel = False):
    #
    deleted_cells = []
    lab_img = make_label(label > vmin)
    cell_names = np.unique(lab_img)
    cell_names = cell_names[cell_names > 0] 
    cells = []
    cells_clean = []
    cell_lab = 0           
    for ncell in cell_names:
        locs = np.where(lab_img == ncell)
        xctr,yctr = np.average(np.array(locs),axis = 1)
        xstd,ystd = np.std(np.array(locs),axis = 1)
        r = np.sqrt(xstd**2 + ystd**2)
          
        
        if r > rmin:
            cells.append([cell_lab,xctr,yctr,r])

            if cell_lab in deleted_labs:
                deleted_cells.append(ncell)   
            else:
                cells_clean.append(cells[-1])
        
            cell_lab +=1
        
        
        else:
            deleted_cells.append(ncell)
            
            
                        
    if cleanlabel:
        for ncell in deleted_cells:
            lab_img[lab_img == ncell] = 0
            
            
    return cells, 1.0*(lab_img > vmin), cells_clean


def decodename(name):
    
    return name.decode('utf-8').split('\\')[-1]

def process_cells(c):
    
    if len(c) == 0:
        c.append([-1,np.nan,np.nan,np.nan])
    
    return c
    
#%%
if __name__ == '__main__':
    
    
    model_path = '//vs01/SBL_DATA/lorenzo/segmentation/brain/models/fulltrain_densetile_{}epochs.hdf5'.format(100)
    rev_path = '//vs01/SBL_DATA/lorenzo/PROJECTS/cell_segmentation/review/'
    
    
    # Alternatively, load the weights directly: model.load_weights(save_model_path)
    model = models.load_model(model_path, custom_objects={'bce_dice_loss': bce_dice_loss,
                                                               'dice_loss': dice_loss})
    
    #%%
    
    ov_df = pd.read_csv(rev_path + 'slices_14-31_2019-07-08.csv', sep = ';', decimal = ',', index_col = 0)
    #%%
    perf_df = ov_df.loc[ov_df.perf]
    #%%    
    noadd_df = ov_df.loc[np.invert(ov_df.perf) * ov_df.noadd]
    
    #%%%
    user = 'rev_ML'
    
    slicenumbers = pd.unique(ov_df.slice)   

    for N_slice in slicenumbers:
        
        #%
        img_dir = '//vs01/SBL_DATA/lorenzo/PROJECTS/cell_segmentation/datasets/slice_{:04d}_tiled'.format(N_slice)
        rev_path_tr = rev_path + 'slice_{:04d}_afterrev_train/'.format(N_slice)
        if not os.path.isdir(rev_path_tr):
            os.makedirs(rev_path_tr)
        rev_path_ma = rev_path + 'slice_{:04d}_afterrev_train_masks/'.format(N_slice)
        if not os.path.isdir(rev_path_ma):
            os.makedirs(rev_path_ma)
        rev_path_ov = rev_path + 'slice_{:04d}_afterrev_overview/'.format(N_slice)
        if not os.path.isdir(rev_path_ov):
            os.makedirs(rev_path_ov)
        
        #%     
        perf_sl = perf_df.loc[perf_df.slice == N_slice]
        noadd_sl = noadd_df.loc[noadd_df.slice == N_slice]
        #%
     
        for df in [perf_sl]:
            
            
            val_filenames = [os.path.join(img_dir, f) for f in df.filename]
            batch_size = 3
        
            img_shape = [256,256]
            
            val_cfg = {
                'resize': [img_shape[0], img_shape[1]],
                'scale': 1 / 255. ,
            }
            val_preprocessing_fn = functools.partial(_augment, **val_cfg)    
                
            N_files = len(val_filenames)    
            print('Loading a dataset of {} files...'.format(N_files))
            
            #%
            val_ds = get_baseline_dataset(val_filenames, 
                                           preproc_fn=val_preprocessing_fn,
                                           batch_size=batch_size,
                                           shuffle=False)    
        
                
            # Let's visualize some of the outputs 
            data_aug_iter = val_ds.make_one_shot_iterator()
            next_element = data_aug_iter.get_next()
                
            #%
            
            rmin = 2
            vmin = 0.5
            
            pred_dir = img_dir + '_prediction'
            if not os.path.isdir(pred_dir):
                os.makedirs(pred_dir)
                    
            # Running next element in our graph will produce a batch of images
            N_batchs = int(N_files/batch_size)
            for i in range(N_batchs):
                batch_of_imgs,batch_of_fnames = tf.keras.backend.get_session().run(next_element)
                predicted_labels = model.predict(batch_of_imgs)
            
                for l in range(batch_size):
                    #
        
                    this_file = decodename(batch_of_fnames[l])
        
                    delcells = df.loc[df.filename == this_file].deleted.values[0][1:-1].split(',')
                    try:
                        delcells = [int(c) for c in delcells]
                    except:
                        delcells = []
        
                    cells,clean_label,clean_cells = name_cells(predicted_labels[l,:,:,0], deleted_labs = delcells, cleanlabel = True)
        
        #            status = []
        #
        #            # Go and check only if there are cells            
        #            if len(cells) > 0:
        #                                
                    plot_empty_vs_ml(batch_of_imgs[l],predicted_labels[l,:,:,0],clean_label,cells,clean_cells, this_file)
        
                    img_out = np.array(batch_of_imgs[l] * 255., dtype = np.uint16)
                    cv2.imwrite(os.path.join(rev_path_tr, this_file), img_out)
            
                    lab_out = np.array( clean_label * 255., dtype = int)
                    cv2.imwrite(os.path.join(rev_path_ma, this_file.replace('.png','_mask.png')),lab_out)  