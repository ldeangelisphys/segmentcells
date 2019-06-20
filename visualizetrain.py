from tensorflow.python.keras import models
from tensorflow.python.keras import losses
import tensorflow as tf
import functools
import os
import matplotlib.pyplot as plt
from skimage.measure import label as make_label
import numpy as np
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
    print(x0,y0)
    print(cells)
    
        
    return np.argmin(d)

def on_button_release(event,img,lab,cells):
    
    if event.button == 1:
        
        cells[-1][3] = np.sqrt( (cells[-1][1] - event.ydata)**2 +
                                (cells[-1][2] - event.xdata)**2)
        
    show_window( event, img, lab, cells)
        

def on_button_press( event, img,lab,cells):
    
    
    # With a left click, add a cell    
    if event.button == 1:

        new_cell = [np.max(np.array(cells)[:,0])+1,
                    event.ydata,
                    event.xdata,
                    10]
    
        cells.append(new_cell)    
        if event.button != 1:
            print('ciao')
        
        
    # For a right click, remove the closest cell    
    if event.button == 3:
        # Decide which spot has to be removed
        idx = find_closest(event.xdata,event.ydata,cells)
        cells.pop(idx)# = np.delete(cells,idx,axis=0)
        
    
    show_window( event, img, lab, cells)

def on_key_press( event ):

    if event.key == 'enter':
        plt.close('all')
    if event.key == 'escape':
        plt.close('all')
    if event.key == 'e':
        plt.close('all')
    if event.key == 'p':
        plt.close('all')
    print(event.key)    
    
    
def plot_empty_vs_ml(img,lab,cells): 

    #    batch_of_imgs = tf.keras.backend.get_session().run(next_element)[0]
#    predicted_labels = model.predict(batch_of_imgs)
#    labels = np.array([predicted_labels,predicted_labels])
#    img,lab= batch_of_imgs[0],labels[:,0,:,:,0]
    fig,ax = plt.subplots(ncols=2,figsize=(20, 10), sharex =True, sharey= True)

    for nplot in range(2):
        plot_back_img(ax[nplot],img)
        

    ax[nplot].contour(lab[nplot,:,:], levels = [0.5], colors = 'red', linestyles = ':')


    plot_circles(ax[nplot],np.array(cells))
#             ax[nplot].imshow(lab_img, cmap = 'hsv', alpha = 0.2)

    ax[1].set_title('ML Segmentation')
    
    
    # Why the lambda?? can I just put on_button_press? TODO
    fig.canvas.mpl_connect( 'button_press_event', lambda event: on_button_press( event, img,lab, cells) )
    fig.canvas.mpl_connect( 'button_release_event', lambda event: on_button_release( event, img,lab, cells) )
    fig.canvas.mpl_connect( 'key_press_event', lambda event : on_key_press( event ) )

    fig.suptitle("Examples of Input Image, Label, and Prediction")
    plt.show()

    return cells

def plot_back_img(ax,img):

    ax.imshow(img[:,:,1]-img[:,:,0], vmax=10, cmap = 'Greys_r', origin = 'lower')
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


def name_cells(label):
    
    lab_img = make_label(label > vmin)
    cell_names = np.unique(lab_img)
    cell_names = cell_names[cell_names > 0] 
    cells = []
    cell_lab = 0           
    for ncell in cell_names:
        locs = np.where(lab_img == ncell)
        xctr,yctr = np.average(np.array(locs),axis = 1)
        xstd,ystd = np.std(np.array(locs),axis = 1)
        r = np.sqrt(xstd**2 + ystd**2)
        if r > rmin:
            cells.append([cell_lab,xctr,yctr,r])
            cell_lab +=1
    
    return cells


#%%
if __name__ == '__main__':
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
    for i in range(1):
        batch_of_imgs = tf.keras.backend.get_session().run(next_element)[0]
        predicted_labels = model.predict(batch_of_imgs)
    #    
        labels = np.array([predicted_labels,predicted_labels])
    
        for l in range(1):
    
            # initialize empty list for cells
            cells = name_cells(labels[1,l,:,:,0])
            
            
    #         plot_man_vs_ml(batch_of_imgs[l],labels[:,l,:,:,0],feat = False)
    #         plot_man_vs_ml(batch_of_imgs[l],labels[:,l,:,:,0], raw = False)
            plot_empty_vs_ml(batch_of_imgs[l],labels[:,l,:,:,0],cells)
    
    
