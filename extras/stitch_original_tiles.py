import cv2
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import os


def make_image_dictionary(images, nrows, ncols):

    img_dict = {nr: {} for nr in range(nrows)}

    for i, fim in enumerate(images):
        img = cv2.imread(rawdata_dir + fim, -1)

        nc = int(i / nrows)
        if nc % 2 == 0:
            nr = i % nrows
        else:
            nr = nrows - 1 - i % nrows

        img_dict[nr][nc] = img

    return img_dict


def align_row(img_dict,nrows,ncols,alch):
    xshifts = []
    nxcut = 50

    row_img = {c:{} for c in [1,2,3]}

    for nr in range(nrows):
        for nc in range(1,ncols):
            xsh = adjust_horizontally(img_dict[alch], nr, nc, ncut=nxcut)
            x_overlap = 0.5 * (
                        img_dict[alch][nr][nc - 1][:, -xsh - nxcut:-nxcut] + img_dict[alch][nr][nc][:, nxcut:xsh + nxcut])
            zero_frac = np.sum(x_overlap < 1) / np.prod(x_overlap.shape)
            if zero_frac > 0.95:
                xsh = 60
            #                 print('nc = {}, not possible to estimate overlap'.format(nc))
            xshifts.append([nc, nr, xsh])

        for nc, _, xsh in xshifts:

            for c in [1,2,3]:

                x_overlap = 0.5 * (img_dict[c][nr][nc - 1][:, -xsh - nxcut:-nxcut] + img_dict[c][nr][nc][:, nxcut:xsh + nxcut])

                if nc == 1:
                    row_img[c][nr] = np.append(img_dict[c][nr][nc - 1][:, :-xsh - nxcut], x_overlap, axis=1)
                else:
                    row_img[c][nr] = np.append(row_img[c][nr][:, :-xsh - nxcut], x_overlap, axis=1)
                row_img[c][nr] = np.append(row_img[c][nr], img_dict[c][nr][nc][:, xsh + nxcut:], axis=1)

    return row_img




def adjust_horizontally(img_dict, nr, nc, ncut=50):
    XX = np.arange(100)
    corr = np.zeros_like(XX, dtype=float)
    for xx in XX[40:]:
        corr[xx], _ = pearsonr(img_dict[nr][nc - 1][:, -xx - ncut:-ncut].ravel(),
                               img_dict[nr][nc][:, ncut:xx + ncut].ravel())

    return np.argmax(corr)

def adjust_rows(row_dict, nr, startx=10000, endx=20000, ncut=10):

    ymax = 150
    xmax = 30
    YY = np.arange(150)
    XX = np.arange(-xmax, xmax)
    corr = np.zeros((ymax, 2 * xmax), dtype=float)
    for yy in YY[90:]:
        for xx in XX:
            corr[yy, xx + xmax], _ = pearsonr(row_dict[nr - 1][-yy - ncut:-ncut, startx:endx].ravel(),
                                              row_dict[nr][ncut:yy + ncut, startx + xx:endx + xx].ravel())
    ysh, xsh = np.where(corr == np.max(corr))
    ysh = ysh[0]
    if xsh[0] == 0:
        print('Warning!')
    xsh = xsh[0] - xmax

    return xsh, ysh



if __name__ == '__main__':
    dataset = '2019_11_25'



    nrows = 14
    ncols = 20

    slices = np.arange(31,159,5)
    slices = np.append(slices,np.arange(1,31,5)[::-1])

    for N_slice in slices:

        rawdata_dir = '/data02/Rodent/Rajeev/171103/Data/Group2/171103_02_05/Twophoton/{}/rawData/New_Demo_2_05-{:04d}/'.format(
            dataset, N_slice)
        channels = [1,2,3]
        alignment_ch = 2
        images = {}
        for c in channels:
            images[c] = np.sort([f for f in os.listdir(rawdata_dir) if f.endswith('{:02d}.tif'.format(c))])

        img_dict = {}
        for c in channels:
            img_dict[c] = make_image_dictionary(images[c],nrows,ncols)
        print('Data Imported')


        #### Align each row ############################################
        row_img = align_row(img_dict,nrows,ncols,alignment_ch)
        print('Rows aligned')


        xst = 10000
        xen = 20000
        yshifts = []
        nycut = 10

        for nr in range(1, nrows):

            xsh, ysh = 0,110
            y_overlap = 0.5 * (
                        row_img[alignment_ch][nr - 1][-ysh - nycut:-nycut, xst:xen] + row_img[alignment_ch][nr][nycut:ysh + nycut, xst + xsh:xen + xsh])
            zero_frac = np.sum(y_overlap < 1) / np.prod(y_overlap.shape)

            if zero_frac < 0.95:
                xsh, ysh = adjust_rows(row_img[alignment_ch], nr, startx=xst,endx=xen,   ncut=nycut)

            yshifts.append([nr, ysh, xsh])
        print('rows adjusted')

        ### Make the full image
        Lx = []
        for nr in row_img[alignment_ch]:
            ly, lx = row_img[alignment_ch][nr].shape
            Lx.append(lx)

        yshifts = np.array(yshifts)
        max_xrowshift = np.max(np.abs(np.cumsum(yshifts[:, 2])))
        min_rowsize = np.min(Lx)
        xst = max_xrowshift + 1
        xen = min_rowsize - max_xrowshift - 1



        for c in channels:
            cumxsh = 0

            for nr, ysh, xsh in yshifts:

                y_overlap = 0.5 * (row_img[c][nr - 1][-ysh - nycut:-nycut, xst + cumxsh:xen + cumxsh] +
                                   row_img[c][nr][nycut:ysh + nycut, xst + cumxsh + xsh:xen + cumxsh + xsh])

                if nr == 1:
                    full_img = np.append(row_img[c][nr - 1][:-ysh - nycut, xst + cumxsh:xen + cumxsh], y_overlap, axis=0)
                else:
                    full_img = np.append(full_img[:-ysh - nycut, :], y_overlap, axis=0)
                full_img = np.append(full_img, row_img[c][nr][ysh + nycut:, xst + cumxsh + xsh:xen + cumxsh + xsh], axis=0)

                cumxsh += xsh


            # Now we have the full_img for one channel
            out_dir = '/data01/lorenzo/PROJECTS/cell_segmentation/datasets/{}/full_slices/'.format(dataset)
            fout = out_dir + '{:04d}_{:02d}.tif'.format(N_slice,c)
            cv2.imwrite(fout, full_img.astype(np.uint16))
