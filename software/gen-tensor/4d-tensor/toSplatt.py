import os, sys

def delete_first_lines(filename, count):
    fin = open(filename, 'r')
    a = fin.readlines()
    fout = open(filename, 'w')
    b = ''.join(a[count:])
    fout.write(b)

if __name__=='__main__':
    #os.system('cp -r hicoo-datasets/ splatt-datasets/')
    filelist = []
    for home, dirs, files in os.walk('./splatt-datasets'):
        for filename in files:
            filelist.append(os.path.join(home, filename))
            #filelist.append(filename)
     
    print(filelist)
    for i in range(len(filelist)):
        delete_first_lines(str(filelist[i]), 2)
