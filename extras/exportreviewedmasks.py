import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from extras.revfunctions import *


dataset = '2019_11_25'
model_path = '/data01/lorenzo/PROJECTS/cell_segmentation/models/fulltrain_densetile_{}epochs.hdf5'.format(100)
rev_path = '/data01/lorenzo/PROJECTS/cell_segmentation/review/{}/'.format(dataset)
data_path = '/data01/lorenzo/PROJECTS/cell_segmentation/datasets/{}/'.format(dataset)
slices = [31]
rev_file = 'slices_{}_2020-07-09.csv'.format('-'.join([str(s) for s in slices]))

df = pd.read_csv(rev_path + rev_file, delimiter=';', decimal = ',')

for i,row in df.iterrows():
    #
    sl = row.slice
    data_dir = data_path + 'slice_{:04d}_tiled/'.format(sl)
    pred_dir = data_path + 'slice_{:04d}_tiled_prediction/'.format(sl)
    mask_dir = data_path + 'slice_{:04d}_tiled_masks/'.format(sl)
    f_img = data_dir + 'slice_{:04d}_tiled/'.format(sl) + row.filename
    f_pre = pred_dir + row.filename.replace('.png','_mask.png')


    #
    if len(row.deleted) > 2:
        to_delete = [int(c) for c in row.deleted[1:-1].split(',')]
    else:
        to_delete = []
    ##
    try:
        pre = cv2.imread(f_pre)[:,:,0] / 255.
    except:
        continue

    lab,cells = fname_cells(pre, vmin = 0.8)
    for d in to_delete:
        try:
            delete_loc = np.where(lab == cells[int(d)][0])
            lab[delete_loc] = 0
        except:
            print('Warning!')
            print('file {}, cell {} not found'.format(row.filename,d))

    # Save
    lab_tile = (lab > 0 ) * 255  # Essential to save the mask in binary way
    f_mask = mask_dir + row.filename.replace('.png','_mask.png')
    cv2.imwrite(f_mask, lab_tile)