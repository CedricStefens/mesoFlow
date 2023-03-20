import fnmatch
import os


def getFilesPath(currdir,ext="tif"): # 
    # compatible linux
    """ retrieve absolute paths of all *.ext files inside directory  """
    temp=[]           
    for file in os.listdir(currdir):
        if fnmatch.fnmatch(file, '*.'+ext):
            temp.append(currdir+"/"+file)
    return temp