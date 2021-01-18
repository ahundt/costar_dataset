import subprocess 
import pathlib 
import time 
from multiprocessing import Pool
import os 
import sys 

N_JOBS = 16

if len(sys.argv) > 1: 
    output_dir = pathlib.Path(sys.argv[1]) 
    assert(output_dir.exists()) 
else:
    output_dir = pathlib.Path(".") 

base_url = "https://archive.org/download/johns_hopkins_costar_dataset/blocks_only/"

train_filename = "costar_block_stacking_dataset_v0.4_blocks_only_success_only_train_files.txt"
dev_filename = "costar_block_stacking_dataset_v0.4_blocks_only_success_only_val_files.txt"
train_url = base_url + train_filename
dev_url = base_url + dev_filename 
if not (output_dir.joinpath(train_filename).exists()):
    print(f"Downloading {train_url} to {output_dir}...") 
    subprocess.Popen(["wget", train_url, "-P", output_dir]).communicate() 
    print(f"Downloaded {train_url} to {output_dir}") 
if not (output_dir.joinpath(dev_filename).exists()):
    print(f"Downloading {dev_url} to {output_dir}...") 
    subprocess.Popen(["wget", dev_url, "-P", output_dir]).communicate() 
    print(f"Downloaded {dev_url} to {output_dir}") 

with open(output_dir.joinpath(train_filename)) as trn_f1,\
     open(output_dir.joinpath(dev_filename)) as val_f1:
    data = [pathlib.Path(x.strip()).name for x in trn_f1.readlines()]
    data += [pathlib.Path(x.strip()).name for x in val_f1.readlines()]

success_files = [pathlib.Path(x).name for x in output_dir.glob("*success.h5f")]

remaining = set(data) - set(success_files) 
def download(filename):
    url =  base_url  + filename
    subprocess.Popen(["wget", url, "-P", output_dir]).communicate() 
    # sleep to avoid overwhelming the server 
    time.sleep(10) 

with Pool(processes=N_JOBS) as pool:
    pool.map(download, remaining)


