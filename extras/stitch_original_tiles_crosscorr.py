import cv2
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import os


def compute_rmat(alignment_stitched, alignment_tile, ix, iy, shiftx, shifty):
    Lshiftx = len(shiftx)
    Lshifty = len(shifty)

    rmat = np.zeros((Lshiftx, Lshifty))

    optimize_tile = alignment_stitched[yst[iy, ix]:yst[iy, ix] + Ty, xst[iy, ix]:xst[iy, ix] + Tx]
    nonzero = np.where(optimize_tile > 0)
    nonzero_frac = len(nonzero[0]) / np.prod(optimize_tile.shape)
    if nonzero_frac < 0.05:
        print('No info in this image' + 20 * '-')

    else:

        for isx, sx in enumerate(shiftx):
            for isy, sy in enumerate(shifty):
                optimize_tile = alignment_stitched[yst[iy, ix] + sy:yst[iy, ix] + Ty + sy, xst[iy, ix] + sx:xst[iy, ix] + Tx + sx]
                nonzero = np.where(optimize_tile > 0)
                new_tile = alignment_tile['{}_{}'.format(ix, iy)]
                r, p = pearsonr(optimize_tile[nonzero].ravel(), new_tile[nonzero].ravel())
                rmat[isx, isy] = r
        rmat[np.isnan(rmat)] = 0

    return rmat


def overlap_tile(stitched, tiles, channels, alignment_ch, ix, iy, xmax, ymax, xst, yst, allxsh, allysh, dirx=1, diry=1):


    print('Working on tile {}_{}'.format(ix, iy))
    if (ix == htx) & (iy == hty):
        shiftx = np.array([0])
        shifty = np.array([0])
    else:
        shiftx = np.arange(-xmax, xmax + 1)
        shifty = np.arange(-ymax, ymax + 1)

    rmat = compute_rmat(stitched[alignment_ch], tiles[alignment_ch], ix, iy, shiftx, shifty)

    maxr = np.max(rmat)
    xopts, yopts = np.where(rmat == maxr)
    xglobs = int(np.average(shiftx[xopts]))
    yglobs = int(np.average(shifty[yopts]))

    for c in channels:
        stitched[c][yst[iy, ix] + yglobs:yst[iy, ix] + Ty + yglobs,
        xst[iy, ix] + xglobs:xst[iy, ix] + Tx + xglobs] = tiles[c]['{}_{}'.format(ix, iy)]
        print('[{},{}],({},{})[{},{}]\tr={:.2f}'.format(shiftx[0], shiftx[-1],
                                                        xglobs, yglobs,
                                                        shifty[0], shifty[-1], maxr))

    if diry == 1:
        if dirx == 1:
            yst[iy:, ix:] += yglobs
            xst[iy:, ix:] += xglobs
        else:
            yst[iy:, :ix] += yglobs
            xst[iy:, :ix] += xglobs
    else:
        if dirx == 1:
            yst[:iy, ix:] += yglobs
            xst[:iy, ix:] += xglobs
        else:
            yst[:iy, :ix] += yglobs
            xst[:iy, ix:] += xglobs

    xst[xst < 0] = 0
    yst[yst < 0] = 0

    allysh[iy, ix] = yglobs
    allxsh[iy, ix] = xglobs

    return



if __name__ == '__main__':

    datasets_info = {
                        '2019_11_25': {
                            'location': '/data02/Rodent/Rajeev/171103/Data/Group2/171103_02_05/Twophoton/2019_11_25/rawData/New_Demo_2_05-{:04d}/',
                            'nrows': 14,
                            'ncols': 20
                    },
                        '2019_02_20': {
                            'location': '/data02/Rodent/Rajeev/171103/3_02/Test_Brain1664_stack/rawData/Test_Brain1664_302_40x2z-{:04d}/',
                            'nrows': 10,
                            'ncols': 16
                    }
    }

    dataset = '2019_11_25'
    # dataset = '2019_02_20'


    nrows = datasets_info[dataset]['nrows']
    ncols = datasets_info[dataset]['ncols']
    ntiles = nrows*ncols

    slices = np.arange(46,53)

    for N_slice in slices:

        rawdata_dir = datasets_info[dataset]['location'].format(N_slice)
        channels = [1,2,3]
        alignment_ch = 1
        images = {}
        for c in channels:
            images[c] = np.sort([f for f in os.listdir(rawdata_dir) if f.endswith('{:02d}.tif'.format(c))])

        cutx, cuty = 50, 1
        tiles = {}
        for c in channels:
            tiles[c] = {}

            for i, fim in enumerate(images[c]):

                nc = int(i / nrows)
                if nc % 2 == 0:
                    nr = i % nrows
                else:
                    nr = nrows - 1 - i % nrows

                img = cv2.imread(rawdata_dir + fim, -1)
                tiles[c]['{}_{}'.format(nc, nr)] = img[cuty:-cuty,cutx:-cutx]
                if i > ntiles:
                    break
        print('Data Imported')

        Ty, Tx = tiles[alignment_ch]['0_0'].shape
        Nx = ncols
        Ny = nrows

        hty = int(nrows / 2)
        htx = int(ncols / 2)

        cols = [n for n in range(ncols)]
        rows = [n for n in range(nrows)]

        stitched = {}
        for c in channels:
            stitched[c] = np.zeros((Ny * Ty, Nx * Tx))

        pred_xsh = 60
        pred_ysh = 130

        xmax = 20
        ymax = 20

        # xst = np.zeros((Ny,Nx),dtype = int)
        yst, xst = np.mgrid[:Ny, :Nx]
        allxsh = np.zeros_like(xst)
        allysh = np.zeros_like(yst)
        yst *= (Ty - pred_ysh)
        xst *= (Tx - pred_xsh)

        for c in channels:
            stitched[c][hty * (Ty - pred_ysh):(hty) * (Ty - pred_ysh) + Ty,
            htx * (Tx - pred_xsh):(htx) * (Tx - pred_xsh) + Tx] = tiles[c]['{}_{}'.format(htx, hty)]

        # down right
        for ix in cols[htx:]:
            for iy in rows[hty:]:
                overlap_tile(stitched, tiles, channels, alignment_ch, ix, iy, xmax, ymax, xst, yst, allxsh, allysh)
        print('Down right done ------------------------')
        # down left
        for ix in cols[:htx][::-1]:
            for iy in rows[hty:]:
                overlap_tile(stitched, tiles, channels, alignment_ch, ix, iy, xmax, ymax, xst, yst, allxsh, allysh, dirx=-1)
        print('Down left done ------------------------')
        # up right
        for ix in cols[htx:]:
            for iy in rows[:hty][::-1]:
                overlap_tile(stitched, tiles, channels, alignment_ch, ix, iy, xmax, ymax, xst, yst, allxsh, allysh, diry=-1)
        print('Up right done ------------------------')
        # up left
        for ix in cols[:htx][::-1]:
            for iy in rows[:hty][::-1]:
                overlap_tile(stitched, tiles, channels, alignment_ch, ix, iy, xmax, ymax, xst, yst, allxsh, allysh, dirx=-1, diry=-1)
        print('Up left done ------------------------')


        for c in channels:
            # Now we have the full_img for one channel
            out_dir = '/data01/lorenzo/PROJECTS/cell_segmentation/datasets/{}/full_slices/'.format(dataset)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            fout = out_dir + '{:04d}_{:02d}.tif'.format(N_slice,c)
            cv2.imwrite(fout, stitched[c].astype(np.uint16))
