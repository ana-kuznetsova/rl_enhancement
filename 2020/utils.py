import os
import sys

def main(dir):
    fnames = os.listdir(dir)
    
    for i, fname in enumerate(fnames):
        src = os.path.join(dir, fname)
        dst = os.path.join(dir, 'Train_'+str(i)+'.wav')
        os.rename(src, dst)

if __name__ == '__main__': 
    main(sys.argv[1])