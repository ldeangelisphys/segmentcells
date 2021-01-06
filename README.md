# C-FOS Cell Segmentation with Deep Learning

This software suite uses deep learnig alghoritms from tensorflow to perform cell segmentation.


![Cells count](https://github.com/ldeangelisphys/segmentcells/blob/master/examples/out.gif)

The specific method used is a tensorflow implementation of the U-net alghorithm described in this [arXiv paper](https://arxiv.org/abs/1505.04597). We tested the tool on a dataset consisting of c-fos fluorescence microscopy images. With additional training the method can be applied to other types of microscopy images.

## Train

Setting up the training requires several preprocessing steps. After these are done the training itself is not a big deal
as it is performed by tensorflow. To make sure that every step is performed correctly it can be usefult to have this
software as a Jupyter notebook.

The notebook *train.ipynb* contains the needed preprocessing steps as well as a few visualizations to check that everything
is going as expected. Additionally it contains some basic features for data augmentation. Finally it contains the training
itself, as well as a simple visualization of its efficiency.


## Visualize the training

To launch the software type

    python visualizetrain.py
    
on your anaconda prompt

### Use the visualize training tool to give feedback and improve further training:

The software highlights the identified cells with red contours. This
is the result of the neural network model that performed the segmentation
on the image. Additionally, there are yellow circles with numbers (labels).
These are the cells that survived the criterion for minimum size and brightness
that are actually going to be labeled and stored.

![Visualize training](https://github.com/ldeangelisphys/segmentcells/blob/master/examples/vistrain.png)

With this software you can review whether these cells are correctly identified
or not. To remove a cell right-click near its center. To add a cell, left-click
on its center and release the click on its closest edge.

After completing the review you can save and end the session by pressing one
of these buttons:

| Button        | Meaning    |
| :--           | :-- |
| E or Escape   | If the image was completely **empty** (black) |
| P or Space    |  If the segmentation was **perfect** and you did not need any correction |
| Enter         |  When you applied **modfications** to the image and you want to confirm |
| S or B        | If the image is **bad** for training (not easy to solve manually) |
| O             |  For something **else** |

## Examples folder

The examples folder contains a pretrained network, trained on manually segmented images like the ones in /exmaples/slice_0016_tiled_masks. Moreover, there are sample images (slice_0016_tiled) and their segmentation (slice_0016_tiled_masks), so that one can try the different features of the software. Note that those images alone are not sufficient to perform a satisfactory training.



