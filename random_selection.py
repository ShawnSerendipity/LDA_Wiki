import os
import random
import shutil

if __name__ == "__main__":

    dirpath = 'C:/Users/Xu(Shawn) Zhang/Work/2013_2016_cleaned/2013_2016_cleaned/training'
    destDirectory = 'C:/Users/Public/LDA_Wiki/10k_training'

    filenames = random.sample(os.listdir(dirpath), 2000)
    for fname in filenames:
        srcpath = os.path.join(dirpath, fname)
        shutil.copy(srcpath, destDirectory)