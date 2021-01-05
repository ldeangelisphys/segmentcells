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
    
def plot_img_cells(img,cells,cleancells,fnm,plot_path): 

 
    fig,ax = plt.subplots(ncols=2,figsize=(20, 10), sharex =True, sharey= True)

    for nplot in range(2):
        plot_back_img(ax[nplot],img)
#    ax[1].contour(lab, levels = [0.5], colors = 'red', linestyles = ':')        
    for nj,tcells in enumerate([cells,cleancells]):
        plot_circles(ax[nj],np.array(tcells))
        try:
            outer = np.ones(tcells.shape)
            outer[:,3] *= outf
            outer *= tcells
            plot_circles(ax[nj],np.array(outer),ls = '--', write_index=False, write_r=False)
        except:
            continue
    #             ax[nplot].imshow(lab_img, cmap = 'hsv', alpha = 0.2)

    ax[1].set_title('ML Segmentation')


    fig.suptitle('Segmentation of {}'.format(fnm))
    plt.savefig(plot_path + fnm)
    plt.close('all')

    return


def compute_cell_contrast(img,cells,outf = 2):

    cimg = img[:,:,1]-img[:,:,0]
    
    XX,YY = np.mgrid[:cimg.shape[0],:cimg.shape[1]]
    mask = np.zeros(cimg.shape)
    
    Rc = []
    
    for i,c in enumerate(cells):
        dist = np.sqrt((XX-c[1])**2+(YY-c[2])**2)
        cmask = dist < c[3]
        
        innval = np.average(cimg[cmask])
        
        cmask = (dist > c[3]) * (dist < outf*c[3])
        
        outval = np.average(cimg[cmask])
        mask += cmask       
        
        rc = innval / outval
        
        Rc.append(rc)
        
    cells = np.append(cells,np.array([Rc]).T, axis =1 )

    return cells   


def plot_back_img(ax,img):

    ax.imshow(img[:,:,1]-img[:,:,0], vmin =0, vmax=10, cmap = 'Greys_r', origin = 'lower')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Corrected Segmentation')              

    
    return
        
def plot_circles(ax,cells,ls = '-', fc = 'none', write_index = True,write_r = True):

    for ncell,cell in enumerate(cells):
    
        circle = plt.Circle((cell[2], cell[1]), cell[3], alpha = 0.5,
                            edgecolor = 'yellow', linestyle = ls,
                            fc = 'none', lw =2)
        ax.add_artist(circle)
 
        if write_index:
            ax.text(cell[2] + cell[3], cell[1]+cell[3],int(cell[0]),horizontalalignment = 'left', verticalalignment = 'top', color ='yellow')
        if write_r:
            ax.text(cell[2], cell[1],'{:.1f}'.format(cell[4]),
                    horizontalalignment = 'center', verticalalignment = 'center',
                    weight= 'bold',color ='red')
            
        
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
    
    dataset = '2019_11_25'

    model_path = '//vs01/SBL_DATA/lorenzo/PROJECTS/cell_segmentation/models/fulltrain_densetile_{}epochs.hdf5'.format(100)
    rev_path = '//vs01/SBL_DATA/lorenzo/PROJECTS/cell_segmentation/review/{}/'.format(dataset)
     
#    # Alternatively, load the weights directly: model.load_weights(save_model_path)
#    model = models.load_model(model_path, custom_objects={'bce_dice_loss': bce_dice_loss,
#                                                               'dice_loss': dice_loss})
    
    #%%
    N_slice = 31

    
    ov_df = pd.read_csv(rev_path + 'slices_{}_2020-07-08.csv'.format(N_slice), sep = ';', decimal = ',')

#    det_df = pd.read_csv(rev_path + 'slices_14-31_2019-07-08.csv', sep = ';', decimal = ',', index_col = 0)


    #%%
#    perf_df = ov_df.loc[ov_df.perf]
#    #%%        
#    noadd_df = ov_df.loc[np.invert(ov_df.perf) * ov_df.noadd]
#    
#    #%%%
#    user = 'rev_ML'
    

#    for N_slice,sl_df in ov_df.groupby('slice'):
    sl_df = ov_df.loc[ov_df.slice == N_slice]
#        rev_df = pd.read_csv(rev_path + 'slice_{:04d}_autoe.csv'.format(N_slice),
#                             sep = ';', decimal = ',', index_col = 0)
    ml_df = (pd
             .read_csv(rev_path + 'slice_{:04d}_MLpred_full.csv'.format(N_slice),
                       sep = ';', decimal = ',')
             .assign(inout_ratio = np.nan)
             .assign(was_deleted = np.nan)
             )
    
    #%%
    img_dir = '//vs01/SBL_DATA/lorenzo/PROJECTS/cell_segmentation/datasets/{}/slice_{:04d}_tiled'.format(dataset,N_slice)
    anal_path = rev_path + 'slice_{:04d}_afterrev_analysis/'.format(N_slice)
    plot_path = anal_path + 'plots/'
    kw_path = plot_path + 'kept_wrong/'
    for fld in [anal_path,plot_path,kw_path]:
        if not os.path.isdir(fld):
            os.makedirs(fld)
    #%%

    for i,row in sl_df.iterrows():

        img = cv2.imread(img_dir + '/' + row.filename, -1)[:,:,::-1] / 255.
        #            
        try:
            deleted = np.array(row.deleted[1:-1].split(','),dtype = int)
        except:
            deleted = []
        
        tc_df = ml_df.loc[ml_df.filename == row.filename]
        cells = np.array([[r['index'],r['x'],r['y'],r['r']] for i,r in tc_df.iterrows()])

        cells = compute_cell_contrast(img,cells,outf = 2)
        
        cleancells = np.array([c for c in cells if c[0] not in deleted])
        
#        plot_img_cells(img,cells,cleancells,row.filename,plot_path)
        
  
        del_ix = [x in deleted for x in range(len(cells))]
        
        up_df = pd.DataFrame(np.array([cells[:,4],del_ix]).T,
                             columns=['inout_ratio','was_deleted'], index = tc_df.index)
        ml_df.update(up_df)      





    #%%
    
    clean_df = (ml_df
                .loc[np.invert(pd.isna(ml_df.was_deleted))]
                [['filename','index','r','inout_ratio','was_deleted','eccentricity']]
                )
    clg_df = clean_df.groupby('was_deleted')
    
    #%%
    iobins = np.arange(0,4.1,0.1)
    iocut = [1.2,4]
    plt.axvline(iocut[0],0,1,c = 'k')
    plt.axvline(iocut[1],0,1,c = 'k')
    for d,df in clg_df:
        
        plt.hist(df['inout_ratio'], label = d, alpha = 0.3, bins = iobins)
    plt.legend()
    
    #%%
    ebins = np.arange(-0.1,1.05,0.05)
    ecut = [0,0.8]
    plt.axvline(ecut[0],0,1,c = 'k')
    plt.axvline(ecut[1],0,1,c = 'k')
    for d,df in clg_df:
        plt.hist(df['eccentricity'], label = d, alpha = 0.3, bins = ebins)
    plt.legend()

    #%%
    rbins = np.arange(0,15.5,0.5)
    rcut = [4,9]
    plt.axvline(rcut[0],0,1,c = 'k')
    plt.axvline(rcut[1],0,1,c = 'k')
    for d,df in clg_df:
        plt.hist(df['r'], label = d, alpha = 0.3, bins = rbins)
    plt.legend()
    
    
    #%%
    clean_df = (clean_df
             .assign(to_delete=lambda d:
                 np.invert(
                 (d.eccentricity > ecut[0])*
                 (d.eccentricity < ecut[1])*
                 (d.inout_ratio > iocut[0])*
                 (d.inout_ratio < iocut[1])*
                 (d.r > rcut[0]) *
                 (d.r < rcut[1])
                 ))
             .assign(was_deleted = lambda d:d.was_deleted==1)
             .assign(del_correct = lambda d:d.to_delete*d.was_deleted)
             .assign(del_wrong = lambda d:d.to_delete*np.invert(d.was_deleted))
             .assign(kept_correct = lambda d:np.invert(d.to_delete)*np.invert(d.was_deleted))
             .assign(kept_wrong = lambda d:np.invert(d.to_delete)*d.was_deleted)
             )
                 
    autodel_df = (clean_df
                  [['del_correct','del_wrong','kept_correct','kept_wrong']]
                  .apply(lambda d : np.sum(d) / len(d) * 100)
                  )

    autodel_df
    #%%
    hit_rate = 100*autodel_df['kept_correct'] / (autodel_df['kept_correct']+autodel_df['del_wrong'])
    fp_rate = 100*autodel_df['kept_wrong'] / (autodel_df['kept_correct']+autodel_df['del_wrong'])
    print(hit_rate,fp_rate)
    #%%
    kw_df = clean_df.loc[clean_df.kept_wrong][['index','filename']]
    
    for ikw,rkw in kw_df.iterrows():

        img = cv2.imread(img_dir + '/' + rkw.filename, -1)[:,:,::-1] / 255.
        #            
        row = ml_df.loc[ikw] 

    #%
        try:
            deleted = np.array(row.deleted[1:-1].split(','),dtype = int)
        except:
            deleted = []
        
        tc_df = ml_df.loc[ml_df.filename == row.filename]
        cells = np.array([[r['index'],r['x'],r['y'],r['r']] for i,r in tc_df.iterrows()])

        cells = compute_cell_contrast(img,cells,outf = 2)
        
        cleancells = np.array([c for c in cells if c[0]!= rkw['index']])
        
        plot_img_cells(img,cells,cleancells,row.filename,kw_path)
        
    
    #%%

    plt.figure(figsize = (10,10))
    cdict = {1:'C1',0:'C0'}
    for d,df in clg_df:
        plt.scatter(df['r'],df['inout_ratio'], s = 10*df['eccentricity'],
                    color = cdict[d], alpha = 0.5)
    plt.xlim(0,10)
    plt.ylim(0.5,3)


    #%
#    plt.figure(figsize = (10,10))

    H = {}
    for d,df in clg_df:
        H[d],_,_  = np.histogram2d(df['inout_ratio'],df['r'],bins = [iobins,rbins])
        
        
    absmax = np.max(np.abs(H[0]-H[1]))
    plt.imshow(H[0]-H[1], cmap = 'PRGn', origin = 'lower',
               vmin=-absmax,vmax=absmax,
               extent = [rbins[0],rbins[-1],iobins[0],iobins[-1]])
    plt.contour(H[0]-H[1],levels = [0],
                extent = [rbins[0],rbins[-1],iobins[0],iobins[-1]])
    
    plt.xlabel('r (pixels)')
    plt.ylabel(r'$I_{in}/I_{out}$')
    #%%

    plt.figure(figsize = (10,10))
    cdict = {1:'C1',0:'C0'}
    for d,df in clg_df:
        plt.plot(df['r'],df['eccentricity'],'o', color = cdict[d], alpha = 0.5)
    plt.xlim(0,10)
    plt.ylim(-0.5,1.5)


    #%
#    plt.figure(figsize = (10,10))

    H = {}
    for d,df in clg_df:
        H[d],_,_  = np.histogram2d(df['eccentricity'],df['r'],bins = [ezbins,rbins])
        
        
    absmax = np.max(np.abs(H[0]-H[1]))
    plt.imshow(H[0]-H[1], cmap = 'PRGn', origin = 'lower',
               vmin=-absmax,vmax=absmax,
               extent = [rbins[0],rbins[-1],ebins[0],ebins[-1]])
    plt.contour(H[0]-H[1],levels = [0],
                extent = [rbins[0],rbins[-1],ebins[0],ebins[-1]])
    
    plt.xlabel('r (pixels)')
    plt.ylabel(r'$I_{in}/I_{out}$')
    
    
    #%%
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111, projection='3d')

    for d,df in clg_df:
        ax.plot(df['r'],df['eccentricity'],df['inout_ratio'],'o',color = cdict[d], alpha = 0.5)
        
    ax.set_xlabel('r (pixels)')
    ax.set_xlim([0,10])
    ax.set_ylabel('eccentricity')
    ax.set_zlabel(r'$I_{in}/I_{out}$')
    
    #%
    H3 = {}
    for d,df in clg_df:
        H3[d],A = numpy.histogramdd((df['r'],df['eccentricity'],df['inout_ratio']),
                                          bins = [rbins,ebins,iobins])
      
    YYY,XXX,ZZZ = np.meshgrid(0.5*(ebins[:-1]+ebins[1:]),
                              0.5*(rbins[:-1]+rbins[1:]),
                              0.5*(iobins[:-1]+iobins[1:]))

    #%%
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111, projection='3d')

    H3diff = H3[0]-H3[1]
    absmax3 = 0.5* np.max(np.abs(H3diff))
    to_plot = np.where(np.abs(H3diff) > 0)
    
    ax.scatter(XXX[to_plot].ravel(),YYY[to_plot].ravel(),ZZZ[to_plot].ravel(),
               c = H3diff[to_plot].ravel(), cmap= 'PRGn', marker = 'o',
               vmax = absmax3, vmin = -absmax3, alpha = 0.6, s = 200)
      
    ax.set_xlabel('r (pixels)')
    ax.set_xlim([0,10])
    ax.set_ylabel('eccentricity')
    ax.set_zlabel(r'$I_{in}/I_{out}$')
    
    