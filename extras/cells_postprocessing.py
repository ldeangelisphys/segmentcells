import numpy as np

def compute_cell_contrast(img, cells, outf=2):
    cimg = img[:, :, 1] - img[:, :, 0]

    XX, YY = np.mgrid[:cimg.shape[0], :cimg.shape[1]]
    mask = np.zeros(cimg.shape)

    Rc = []

    for i, c in enumerate(cells):
        dist = np.sqrt((XX - c[1]) ** 2 + (YY - c[2]) ** 2)
        cmask = dist < c[3]

        innval = np.average(cimg[cmask])

        cmask = (dist > c[3]) * (dist < outf * c[3])

        outval = np.average(cimg[cmask])
        mask += cmask

        rc = innval / outval

        Rc.append(rc)

    cells = np.append(cells, np.array([Rc]).T, axis=1)

    return cells