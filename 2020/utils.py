import os

def main():
    fnames = os.listdir()
    
    for i, fname in enumerate(fnames):
        os.rename(fname, 'Train_'+str(i)+'.wav')

if __name__ == '__main__': 
    main()