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


def define_tiles(Lx, Ly, Lt, overlap=32):
    Nx = int(Lx / (Lt - overlap)) + 1
    Ny = int(Ly / (Lt - overlap)) + 1
    edges = []

    plt.figure(figsize=(10, 10))
    plt.fill_between([0, Lx], [Ly, Ly], color='gray', alpha=0.1)

    for nx in range(Nx):
        for ny in range(Ny):

            e = [nx * (Lt - overlap), nx * (Lt - overlap) + Lt, ny * (Lt - overlap), ny * (Lt - overlap) + Lt]
            if e[1] > Lx:
                e[:2] = [Lx - Lt, Lx]
            if e[3] > Ly:
                e[2:] = [Ly - Lt, Ly]
            edges.append(e)

            plt.fill_between([e[0], e[1]],
                             [e[3], e[3]],
                             [e[2], e[2]], color='C4',
                             alpha=0.3, linestyle='-', linewidth=2)

    plt.axis('off')
    plt.title('Original image ({}x{}) and its tiles ({}x{})'.format(Lx, Ly, Lt, Lt))

    #    plt.savefig('U:/PROJECTS/cell_segmentation/tiling/procedure.png', dpi = 300)
    #    plt.close('all')
    return edges

    
def make_tiles(raw_fld,out_root,Lt,overlap,have_labels = False, lab_fld = None, start_from = 0):
    

    
    raw_files = [f.split('_01.t')[0] for f in os.listdir(raw_fld) if f.endswith('_01.tif')]
    N_files = len(raw_files)
    print(raw_files)

    # #############################################
    # if have_labels:
    #     lab_files = [f for f in os.listdir(lab_fld + 'PixelLabelData') if f.endswith('.png')]
    #     # Build a converter for the label names
    #     img2lab = {}
    #
    #     df = pd.read_csv(lab_folder + 'PixelLabelData/labelconversion.csv').dropna()
    #
    #     for i in df.index:
    #         labfname = df['PixelLabelData'][i].split('\\')[-1]
    #         imgfname = df['Var1'][i].split('\\')[-1]
    #         img2lab[imgfname.split('01_SB')[0]] = labfname
    #
    #     out_fld_lab = out_fld + '_masks'
    #     if not os.path.isdir(out_fld_lab):
    #         os.makedirs(out_fld_lab)
    #
    #     # To make sure I am only looking at files with labels in them
    #     raw_files = [f for f in img2lab.keys()]
    # #############################################


    # Color coding for input and output layers
    col_in = {'red':2,'green':1,'blue':3}
    col_out = {'red':2,'green':1,'blue':0} ### STUPID BGR WAY OF CV2
    
    # def process_original_images(filenames,nameconv,raw_fld,out_fld,N_tiles,ch_shp = (1664,1664)):


    for i,ifile in enumerate(raw_files):


        null_tiles = []

        N_slice = int(ifile)

        if N_slice >= start_from:

            out_fld = out_root + 'slice_{}_tiled/'.format(ifile)
            if not os.path.isdir(out_fld):
                os.makedirs(out_fld)


            # #############################################
            # if have_labels:
            #     label = cv2.imread(os.path.join(lab_fld + 'PixelLabelData', img2lab[ifile]), -1)
            # #############################################

            # Read and incorporate the different layers of the input
            for ic,c in enumerate(col_in):

                fname = ifile + '_{:02d}.tif'.format(col_in[c])

                path_to_img = os.path.join(raw_fld, fname)

                this_layer = cv2.imread(path_to_img, -1)
                if ic == 0:
                    Lx,Ly = this_layer.shape
                    image = np.empty((Lx,Ly,3), dtype = 'uint16')

                image[:,:,col_out[c]] = this_layer



            # Define edges and save it as independent tiles
            edges = define_tiles(Lx,Ly,Lt,overlap)
            N_tiles = len(edges)
            for it,e in enumerate(edges):

                cx,cy = np.reshape(e,(2,2)).mean(axis = 1).astype(int)

                fout = ifile + '_{:05d}_{:05d}_overlap.png'.format(cx,cy)

                tile = image[e[0]:e[1],e[2]:e[3], :]
                zero_frac = np.sum(tile < 2) / np.prod(tile.shape)
                if zero_frac < 0.99:

                    # Both for the images
                    cv2.imwrite(os.path.join(out_fld, fout),tile)


                    # #############################################
                    # if have_labels:
                    #     lab_tile = label[e[0]:e[1],e[2]:e[3]] * 255 # Essential to save the mask in binary way
                    #
                    #     if np.max(lab_tile) > 200:
                    #         fout = ifile + '{:03d}_{:03d}_dense_mask.png'.format(cx,cy)
                    #     else:
                    #         fout = ifile + '{:03d}_{:03d}_dense_mask_[empty].png'.format(cx,cy)
                    #
                    #     cv2.imwrite(os.path.join(out_fld_lab, fout), lab_tile)
                    # #############################################

                else:
                    null_tiles.append(fout)


                progress_bar(it+1,N_tiles)
            print('\n' + '='*25 + ' Slice {:04d} done '.format(N_slice) + '='*25)
            print('{} tiles skipped out of {} (which is {:.0f}%)'.format(
                len(null_tiles),N_tiles,100*len(null_tiles)/N_tiles))
            print('='*67)


    return

#%%


if __name__ == '__main__':

    tile_size = 256
    overlap = 32
    

    dataset = '2019_11_25'
    # dataset = '2019_02_20'

    raw_fld = '/data01/lorenzo/PROJECTS/cell_segmentation/datasets/{}/full_slices/'.format(dataset)
    out_root = '/data01/lorenzo/PROJECTS/cell_segmentation/datasets/{}/tiled_slices/'.format(dataset)
        #
    image_content = make_tiles(raw_fld,out_root,tile_size,overlap,start_from=35)
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
