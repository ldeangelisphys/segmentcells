# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:23:24 2020

@author: deangelis
"""
import pandas as pd
from datetime import date

dataset = '2019_11_25'

model_path = '/data01/lorenzo/PROJECTS/cell_segmentation/models/fulltrain_densetile_{}epochs.hdf5'.format(100)
rev_path = '/data01/lorenzo/PROJECTS/cell_segmentation/review/{}/'.format(dataset)

today = date.today().strftime("%Y-%m-%d")
#%%

slices = [31]

N_slice = slices[0]


ov_df = pd.read_csv(rev_path + 'slice_{:04d}_autoe.csv'.format(N_slice), sep = ';', decimal = ',')
ml_df = (pd
         .read_csv(rev_path + 'slice_{:04d}_MLpred_full.csv'.format(N_slice),
                   sep = ';', decimal = ',')
         )

#%%

reorg_df = pd.DataFrame()


for full_fn in ov_df.filename.unique():

    full_fn = '/data01/lorenzo/PROJECTS/cell_segmentation/datasets/2019_11_25/slice_0031_tiled/11082019-1749-8476_128_1536_dense.png'
    fn = full_fn.split('/')[-1]
    labeled_ML   = list(ml_df.loc[ml_df.filename == fn]['index'])
    labeled_user = list(ov_df.loc[ov_df.filename == full_fn]['index'])
    print(ov_df.loc[ov_df.filename == full_fn]['index'])
    print(labeled_user)
    break


    # if (len(labeled_user) == 1)&(len(labeled_ML) == 1)&(labeled_ML[0] == -1):
    #     continue
    # else:
    #
    #     untouched = [c for c in labeled_ML if c in labeled_user and c>=0]
    #     deleted = [c for c in labeled_ML if c not in labeled_user and c>=0]
    #     added = [c for c in labeled_user if c not in labeled_ML and c>=0]
    #
    #     nu = len(untouched)
    #     nd = len(deleted)
    #     na = len(added)
    #
    #     entry = {'filename':[fn.split('/')[-1]],
    #              'slice':[N_slice],
    #              'untouched':[untouched],
    #              'deleted':[ deleted],
    #              'added': [added],
    #              'nu':[nu],
    #              'nd':[nd],
    #              'na':[na],
    #              'ncells':[nu+na+nd],
    #              'perf':[(na==0)&(nd==0)],
    #              'noadd':[(na==0)],
    #              'idle':[1         ]
    #             }
    #
    #     reorg_df = reorg_df.append(pd.DataFrame(entry))
    #
    # print(fn)
    #
    # #%%
    #
    # reorg_df.to_csv(rev_path + 'slices_{}_{}.csv'.format('-'.join([str(s) for s in slices]),today),
    #                 sep = ';', decimal = ',', index = False)
    #
    #
    #
    # #break