import cv2 # to install in suite2p environment: conda install -c conda-forge opencv
import numpy as np
import os
import tifffile
from past.utils import old_div
from tqdm import tqdm
from suite2p.io.tiff import open_tiff
from tifffile import imread
import re
from ScanImageTiffReader import ScanImageTiffReader
from mesoFlow.IO import getFilesPath

def saveMovie(movie,
             file_name,
             to32=False,
             order='F',
             imagej=False,
             bigtiff=True, 
             compress=0,
             q_max=99.75,
             q_min=1):
        """adapted from caiman github"""
        """
        Save the timeseries in single precision. Supported formats include
        TIFF,  AVI . from caiman github.
        Args:
            file_name: str
                name of file. Possible formats are tif, avi, npz, mmap and hdf5
            to32: Bool
                whether to transform to 32 bits
            order: 'F' or 'C'
                C or Fortran order 
            q_max, q_min: float in [0, 100]
                percentile for maximum/minimum clipping value if saving as avi
                (If set to None, no automatic scaling to the dynamic range [0, 255] is performed)
 
        """
        name, extension = os.path.splitext(file_name)[:2]
        extension = extension.lower()

        if extension in ['.tif', '.tiff', '.btf']:
            with tifffile.TiffWriter(file_name, bigtiff=bigtiff, imagej=imagej) as tif:
                for i in range(movie.shape[0]):
                    if i % 200 == 0 and i != 0:
                        print(str(i) + ' frames saved')

                    curfr = movie[i].copy()
                    if to32 and not ('float32' in str(movie.dtype)):
                        curfr = curfr.astype(np.float32)
                    tif.save(curfr)

        elif extension == '.avi':
            codec = None
            try:
                codec = cv2.FOURCC('I', 'Y', 'U', 'V')
            except AttributeError:
                codec = cv2.VideoWriter_fourcc(*'IYUV')
            if q_max is None or q_min is None:
                data = movie.astype(np.uint8)
            else:
                if q_max < 100:
                    maxmov = np.nanpercentile(movie[::max(1, len(movie) // 100)], q_max)
                else:
                    maxmov = np.nanmax(movie)
                if q_min > 0:
                    minmov = np.nanpercentile(movie[::max(1, len(movie) // 100)], q_min)
                else:
                    minmov = np.nanmin(movie)
                data = 255 * (movie - minmov) / (maxmov - minmov)
                np.clip(data, 0, 255, data)
                data = data.astype(np.uint8)
                
            y, x = data[0].shape
            vw = cv2.VideoWriter(file_name, codec, movie.fr, (x, y), isColor=True)
            for d in data:
                vw.write(cv2.cvtColor(d, cv2.COLOR_GRAY2BGR))
            vw.release()

def resizeMovie(movie, fx=1, fy=1, fz=1, interpolation=cv2.INTER_AREA):
    """adapted from caiman github"""
    T, d1, d2 = movie.shape
    d = d1 * d2
    elm = d * T
    max_els = 2**31 - 1
    if elm > max_els:
        chunk_size = old_div((max_els), d)
        new_m:List = []
        for chunk in range(0, T, chunk_size):
            m_tmp = movie[chunk:np.minimum(chunk + chunk_size, T)].copy()
            m_tmp = resizeMovie(m_tmp,fx=fx, fy=fy, fz=fz,
                                 interpolation=interpolation)
            if len(new_m) == 0:
                new_m = m_tmp
            else:
                new_m = np.concatenate([new_m, m_tmp], axis=0)
            print(new_m.shape)
        return new_m
    else:
        if fx != 1 or fy != 1:
            t, h, w = movie.shape
            newshape = (int(w * fy), int(h * fx))
            mov = []
            for frame in movie:
                mov.append(cv2.resize(frame, newshape, fx=fx,
                                      fy=fy, interpolation=interpolation))
            movie = np.asarray(mov)
        if fz != 1:
            t, h, w = movie.shape
            movie = np.reshape(movie, (t, h * w))
            mov = cv2.resize(movie, (h * w, int(fz * t)),
                             fx=1, fy=fz, interpolation=interpolation)
            mov = np.reshape(mov, (np.maximum(1, int(fz * t)), h, w))
    del movie
    mov = np.asarray(mov)
    return mov
	
	
def generateDownsampledRawMovie(subdirs=None,fxy=0.5,fz=0.1,saveConcatenated=False):
    """ generate downsample movie for each group of tiff files present in the subdirs
        e.g. select A/X and A/Y -> A/X/[tiffs,..],A/Y/[tiffs,...] -> A/X_ds.tif, A/Y_ds.tif 
        Args:
            subdirs: list of string
                        list of subdir path, if not provided, dialog box to select them
            fxy: float 
                downsampling ratio in x and y
            fz: float 
                downsampling ratio in z
            saveConcatenated: boolean
                                to additionally save concatenation of all downsized movies"""
    
    fx,fy=fxy,fxy
    
    if subdirs is None:
        return 0
         
    path=('/').join(subdirs[0].split('/')[:-1])
#     print(path)
    for s in subdirs:
        name=s.split('/')[-1]
        print(path+"/"+name)
        r=getFilesPath(s)

        c=0
        m=resizeMovie(imread(r[0]),fx=fx,fy=fy,fz=fz)
        print(c+1," / ", len(r))
        for i in r[1:]:
            c+=1
            print(c+1," / ", len(r))
            m=np.concatenate((m,resizeMovie(imread(i),fx=fx,fy=fy,fz=fz)),axis=0)
        print(m.shape,m.dtype)
 
        if not saveConcatenated:
            saveMovie(m,path+"/"+name+"_ds.tiff") 
            print(m.shape,m.dtype)
        
        if saveConcatenated:
            if s == subdirs[0]:
                mm=m
            elif s== subdirs[-1]:
                mm=np.concatenate((mm,m))
                saveMovie(mm,path+"/raw_concatenated_ds.tiff") 
            else:
                mm=np.concatenate((mm,m))
				
def readSuite2pTiff(fname):
    return open_tiff(fname,False)[0].data()

def concatenateTiffs(dirpath,name="concatenated",deleteTiffs=True):
    fnames=getFilesPath(dirpath)
    fname=dirpath+"/"+name+".tif"
    m=[]
    for f in fnames:
        m+=[readSuite2pTiff(f)]
        if deleteTiffs:
            os.remove(f)
    m=np.concatenate(m,axis=0)
    saveMovie(m,fname)
    return fname
	
def concatenateTiffs_suite2p(dirpath,name="concatenated",deleteTiffs=True):
    fnames=getFilesPath(dirpath)
    fname=dirpath+"/"+name+".tif"
    m=[]
    for f in fnames:
        m+=[readSuite2pTiff(f)]
        if deleteTiffs:
            os.remove(f)
    m=np.concatenate(m,axis=0)
    saveMovie(m,fname)
    return fname
    

def downsampleTiffs(dirpath,fxfyfz=(0.5,0.5,0.1)):
    #downsampled tiff and overwrite them
    fx,fy,fz=fxfyfz
    fnames=getFilesPath(dirpath)
    m=[]
    for f in tqdm(fnames):
        saveMovie(resizeMovie(readSuite2pTiff(f), fx=fx, fy=fy, fz=fz),f)
        
def getFpsScanImage(fname):
    # get fps from scanimage tiff metadata
    fps=float(re.findall("\d+\.\d+", ScanImageTiffReader(fname).metadata().split("scanVolumeRate")[1])[0])
    return fps