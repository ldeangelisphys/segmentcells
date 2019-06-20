from tensorflow.python.keras import models
from tensorflow.python.keras import losses
import tensorflow as tf
import functools
import os
import matplotlib.pyplot as plt
from skimage.measure import label as make_label
import numpy as np
#%%


def plot_empty_vs_ml(img,lab,raw = True, feat = True): 

    fig,ax = plt.subplots(ncols=2,figsize=(20, 10), sharex =True, sharey= True)

    for nplot in range(2):
        ax[nplot].imshow(img[:,:,1]-img[:,:,0], vmax=10, cmap = 'Greys_r', origin = 'lower')
        ax[nplot].set_xticks([])
        ax[nplot].set_yticks([])
        
    if raw:
        ax[nplot].contour(lab[nplot,:,:], levels = [0.5], colors = 'red', linestyles = ':')
        cells = None

    if feat:
        # With blob log (deprecated)
        #Detect the cells in the picture
#             cells = blob_log(lab[nplot,:,:] > vmin)
        # Filter out the ones with radius smaller than rmin
        # With skimage label
        # produce a mask with different numbers for different areas
        lab_img = make_label(lab[nplot,:,:] > vmin)
        cells = []
        cell_names = np.unique(lab_img)
        cell_names = cell_names[cell_names > 0]            
        for ncell in cell_names:
            locs = np.where(lab_img == ncell)
            xctr,yctr = np.average(np.array(locs),axis = 1)
            xstd,ystd = np.std(np.array(locs),axis = 1)
            r = np.sqrt(xstd**2 + ystd**2)
            if r > rmin:
                cells.append([xctr,yctr,r])
        cells = np.array(cells)
        plot_circles(ax[nplot],cells)
#             ax[nplot].imshow(lab_img, cmap = 'hsv', alpha = 0.2)

    ax[0].set_title('Manual Segmentation')              
    ax[1].set_title('ML Segmentation')

    fig.suptitle("Examples of Input Image, Label, and Prediction")
    plt.show()
    
    return cells
        
def plot_circles(ax,cells):

    for ncell,cell in enumerate(cells):
    
        circle = plt.Circle((cell[1], cell[0]), cell[2], alpha = 0.5, edgecolor = 'yellow', fc = 'none', lw =2)
        ax.add_artist(circle)
        ax.text(cell[1] + cell[2], cell[0]+cell[2],ncell,horizontalalignment = 'left', verticalalignment = 'top', color ='yellow')
        
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
  dataset = tf.data.Dataset.from_tensor_slices(filenames)
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

def _process_pathnames(fname):
    # We map this function onto each pathname pair  
    img_str = tf.read_file(fname)
    img = tf.image.decode_png(img_str, channels = 3, dtype = tf.dtypes.uint16)
    
    return img

def _augment(img,
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
  img = tf.to_float(img) * scale 
  return img,


#%%

model_path = 'U:/segmentation/brain/models/fulltrain_densetile_{}epochs.hdf5'.format(100)

# Alternatively, load the weights directly: model.load_weights(save_model_path)
model = models.load_model(model_path, custom_objects={'bce_dice_loss': bce_dice_loss,
                                                           'dice_loss': dice_loss})
    
#%% 
N_slice = 14
img_dir = 'U:/PROJECTS/cell_segmentation/datasets/slice{}_tiled'.format(N_slice)
fnames = [f for f in os.listdir(img_dir) if f.endswith('dense.png')]
val_filenames = [os.path.join(img_dir, f) for f in fnames]
batch_size = 3
#%%
img_shape = [256,256]

val_cfg = {
    'resize': [img_shape[0], img_shape[1]],
    'scale': 1 / 255. ,
}
val_preprocessing_fn = functools.partial(_augment, **val_cfg)    
    
val_ds = get_baseline_dataset(val_filenames, 
                               preproc_fn=val_preprocessing_fn,
                               batch_size=batch_size,
                               shuffle=True)    



   
    
# Let's visualize some of the outputs 
data_aug_iter = val_ds.make_one_shot_iterator()
next_element = data_aug_iter.get_next()
    
#%%

rmin = 2
vmin = 0.5

pred_dir = img_dir + '_prediction'
if not os.path.isdir(pred_dir):
    os.makedirs(pred_dir)


# Running next element in our graph will produce a batch of images
N_batchs = 1
for i in range(6):
    batch_of_imgs = tf.keras.backend.get_session().run(next_element)[0]
    predicted_labels = model.predict(batch_of_imgs)
    
    labels = np.array([predicted_labels,predicted_labels])

    for l in range(batch_size):
        
        
#         plot_man_vs_ml(batch_of_imgs[l],labels[:,l,:,:,0],feat = False)
#         plot_man_vs_ml(batch_of_imgs[l],labels[:,l,:,:,0], raw = False)
        plot_empty_vs_ml(batch_of_imgs[l],labels[:,l,:,:,0])


