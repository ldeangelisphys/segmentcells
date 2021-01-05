import os
import pandas as pd
import numpy as np


if __name__ == '__main__':
    
    
    
    rev_path = '//vs01/SBL_DATA/lorenzo/PROJECTS/cell_segmentation/review/'
    
    fnames = [f for f in os.listdir(rev_path) if f.endswith('autoe.csv')]
    
    ff = fnames[0]
    
    df = pd.read_csv(os.path.join(rev_path,ff), sep = ';', decimal = ',')

    #%%
    wcells = (df.loc[df['index'] >= 0])
    
    counted = len(wcells)
    print('{} cells counted'.format(counted))

    (wcells
     .groupby('filename')
     .apply(lambda d: pd.Series({
             'number of cells':len(d),
             'status': d['exit_status']
             }))
     .reset_index()
     )
    
