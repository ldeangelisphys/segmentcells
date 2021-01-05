from skimage.measure import label
from skimage.measure import regionprops
import numpy as np

def fname_cells(img, vmin = 0.8, rmin = 2):
    lab_img = label(img > vmin)
    Lprop = regionprops(lab_img)
    cell_names = np.unique(lab_img)
    cell_names = cell_names[cell_names > 0]
    cells = {}
    cell_lab = 0
    for ncell in cell_names:
        locs = np.where(lab_img == ncell)
        xctr, yctr = np.average(np.array(locs), axis=1)
        xstd, ystd = np.std(np.array(locs), axis=1)
        r = np.sqrt(xstd ** 2 + ystd ** 2)
        ecc = Lprop[ncell - 1].eccentricity
        if r > rmin:
            cells[cell_lab] = [ncell, xctr, yctr, r, ecc]
            cell_lab += 1
        else:
            lab_img[locs] = 0

    return lab_img,cells