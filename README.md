# C-FOS Cell Segmentation with Deep Learning

![Cells count](https://github.com/ldeangelisphys/segmentcells/blob/master/examples/out.gif)

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
