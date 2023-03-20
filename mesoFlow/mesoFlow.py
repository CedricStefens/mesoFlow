from mesoFlow.IO import getFilesPath
import suite2p

def run_suite2p(ops,directory_path=None):
    if directory_path is not None:
        fnames=getFilesPath(directory_path) # paths to tif files
        ops['data_path']=("/").join(fnames[0].split("/")[:-1])+"/"
        ops['save_folder']=ops['data_path']+"suite2p"
        ops['fast_disk']=ops['data_path']
        ops['tiff_list']=fnames
    ops_reg=suite2p.run_s2p(ops=ops)
    return ops_reg