import matplotlib.pyplot as plt
import numpy as np
import sys, os, cv2
import pandas as pd
#%%


def progress_bar(n,nmax):
    
    perc = 100*(n/nmax)
    prog = int(perc/5)
    
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[%-20s] %d%%" % ('='*prog, perc))
    sys.stdout.flush()

    return


def define_tiles(L,Ltile,check_edges = True):

    ntiles = int(L / Ltile)

    edges = []
    
    plt.figure()
    plt.fill_between([0,L],[L,L], color = 'gray', alpha = 0.1)
    
    for nx in range(ntiles):
        for ny in range(ntiles):
            
            edges.append([nx*Ltile,(nx+1)*Ltile,ny*Ltile,(ny+1)*Ltile])
            
            plt.fill_between([nx*Ltile,(nx+1)*Ltile],
                             [(ny+1)*Ltile,(ny+1)*Ltile],
                             [(ny)*Ltile,(ny)*Ltile], color = 'C4',
                              alpha = 0.3, linestyle = '-', linewidth = 2)
            
    
    for nx in range(ntiles):
        for ny in range(ntiles):
            plt.fill_between([L-(nx+1)*Ltile,L-nx*Ltile],
                             [L-(ny)*Ltile,L-(ny)*Ltile],
                             [L-(ny+1)*Ltile,L-(ny+1)*Ltile], color = 'C8',
                              alpha = 0.3, linestyle = '-', linewidth = 2)
    
            edges.append([L-(nx+1)*Ltile,L-nx*Ltile,L-(ny+1)*Ltile,L-ny*Ltile])
    
            
    plt.fill_between([L-Ltile,L],[Ltile,Ltile],[0,0], color = 'C2',
                     alpha = 0.3, linestyle = '-', linewidth = 2)
    
    plt.fill_between([0,Ltile],[L,L],[L-Ltile,L-Ltile], color = 'C2',
                     alpha = 0.3, linestyle = '-', linewidth = 2)
    
    edges.append([0,Ltile,L-Ltile,L])
    edges.append([L-Ltile,L,0,Ltile])
    
    if check_edges:
        
        for e in edges:
            plt.fill_between([e[0],e[1]],[e[3],e[3]],[e[2],e[2]],
                         alpha = 0.2, color = 'black', linewidth = 0.2)
    
    
    plt.axis('off')
    plt.title('Original image ({}x{}) and its tiles ({}x{})'.format(L,L,Ltile,Ltile))
    
#    plt.savefig('U:/PROJECTS/cell_segmentation/tiling/procedure.png', dpi = 300)
#    plt.close('all')
    return edges

    
def make_tiles(raw_fld,out_fld,edges,have_labels = False, lab_fld = None):
    

    
    raw_files = [f.split('01.t')[0] for f in os.listdir(raw_fld) if f.endswith('01.tif')]
    N_files = len(raw_files)
    N_tiles = len(edges)
 
    #############################################
    if have_labels:
        lab_files = [f for f in os.listdir(lab_fld + 'PixelLabelData') if f.endswith('.png')]
        # Build a converter for the label names
        img2lab = {}
        
        df = pd.read_csv(lab_folder + 'PixelLabelData/labelconversion.csv').dropna()
        
        for i in df.index:
            labfname = df['PixelLabelData'][i].split('\\')[-1]
            imgfname = df['Var1'][i].split('\\')[-1]
            img2lab[imgfname.split('01_SB')[0]] = labfname
        
        out_fld_lab = out_fld + '_masks'
        if not os.path.isdir(out_fld_lab):
            os.makedirs(out_fld_lab)
            
        # To make sure I am only looking at files with labels in them
        raw_files = [f for f in img2lab.keys()]
    #############################################  
    
    # Shape of the input figure
    ch_shp = (image_size,image_size)
    
    # Color coding for input and output layers
    col_in = {'red':2,'green':1,'blue':3}
    col_out = {'red':2,'green':1,'blue':0} ### STUPID BGR WAY OF CV2
    
    # def process_original_images(filenames,nameconv,raw_fld,out_fld,N_tiles,ch_shp = (1664,1664)):
    if not os.path.isdir(out_fld):
        os.makedirs(out_fld)
  
    im_cont = {}
    null_tiles = []
    
    for i,ifile in enumerate(raw_files):
            
        image = np.empty((ch_shp[0],ch_shp[1],3), dtype = 'uint16')

        #############################################        
        if have_labels:
            label = cv2.imread(os.path.join(lab_fld + 'PixelLabelData', img2lab[ifile]), -1)      
        #############################################
    
        # Read and incorporate the different layers of the input
        for c in col_in:
    
            fname = ifile + '{:02d}.tif'.format(col_in[c])
    
            path_to_img = os.path.join(raw_fld, fname)
    
            image[:,:,col_out[c]] = cv2.imread(path_to_img, -1)
    
        # Save it as independent tiles
        for e in edges:
            
            cx,cy = np.reshape(e,(2,2)).mean(axis = 1).astype(int)
            
            fout = ifile + '{:03d}_{:03d}_dense.png'.format(cx,cy)
            
            tile = image[e[0]:e[1],e[2]:e[3], :]
            
            if tile.any() > 0:
                    
                # Both for the images
                cv2.imwrite(os.path.join(out_fld, fout),tile)
                
                
                #############################################        
                if have_labels:
                    lab_tile = label[e[0]:e[1],e[2]:e[3]] * 255 # Essential to save the mask in binary way
                    
                    if np.max(lab_tile) > 200:
                        fout = ifile + '{:03d}_{:03d}_dense_mask.png'.format(cx,cy)
                    else:
                        fout = ifile + '{:03d}_{:03d}_dense_mask_[empty].png'.format(cx,cy)
                        
                    cv2.imwrite(os.path.join(out_fld_lab, fout), lab_tile)
                ############################################# 
                
            else:
                null_tiles.append(fout)
                

        progress_bar(i+1,N_files)
        
    return 

#%%


if __name__ == '__main__':

    image_size = 1664
    tile_size = 256
    
    tile_edges = define_tiles(image_size,tile_size)

    dataset = '2019_11_25'

    N_slice = 17
    raw_fld = '/data02/Rodent/Rajeev/171103/Data/Group2/171103_02_05/Twophoton/{}/rawData/New_Demo_2_05-{:04d}'.format(dataset,N_slice)
    out_fld = '/data01/lorenzo/PROJECTS/cell_segmentation/datasets/{}/slice_{:04}_tiled'.format(dataset,N_slice)
    #

    image_content = make_tiles(raw_fld,out_fld,tile_edges)
    #%%
#
#     N_slice = 16
#     raw_fld = 'V:/Rodent/Rajeev/171103/3_02/Test_Brain1664_stack/rawData/Test_Brain1664_302_40x2z-{:04d}'.format(N_slice)
#     out_fld = 'U:/PROJECTS/cell_segmentation/datasets/slice_{:04}_tiled'.format(N_slice)
#     labels_folder = 'Z:/cFos-Labelling/Slice{}/'.format(N_slice)
#
#     image_content = make_tiles(raw_fld,out_fld,tile_edges,have_labels = True, lab_fld = labels_folder)
#
# #%%
#
#
