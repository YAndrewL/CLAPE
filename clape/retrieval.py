# -*- coding: utf-8 -*-
'''
@File   :  retrieval.py
@Time   :  2024/08/17 19:09
@Author :  Yufan Liu
@Desc   :  Model and training data from Github or somewhere
'''

import requests
import os


def download(url, local_file):
    with requests.get(url=url, stream=True) as response:
        response.raise_for_status()
        with open(local_file, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192): 
                file.write(chunk)
        print(f"{url} is sucessfully download @ {local_file}")


def download_files(path, 
                   file_type,
                   data_type='all'):
    valid_types = ['DNA', 'RNA', 'AB', 'all']
    if data_type not in valid_types:
        raise KeyError(f"data_type should be in {valid_types}")
    if not os.path.exists(path):
        os.makedirs(path)
    
    if file_type == 'weight':
        urls = [f"https://github.com/YAndrewL/CLAPE/blob/main/weights/{d}.pth" for d in valid_types[:-1]]
        locals = [os.path.join(path, d + ".pth") for d in valid_types[:-1]]
    elif file_type == 'data':
        urls = [f"https://github.com/YAndrewL/CLAPE/blob/main/raw_data/{d}.tar.gz" for d in valid_types[:-1]]
        locals = [os.path.join(path, d + ".tar.gz") for d in valid_types[:-1]]
    else:
        raise NotImplementedError()
    
    if data_type == 'all':
        [download(u, l) for u, l in zip(urls, locals)]
    else:
        idx = valid_types.index(data_type)
        download(urls[idx], locals[idx])
          
def get_data(path, data_type):
    download_files(path, 'data', data_type)

def get_weights(path, data_type):
    download_files(path, 'weights', data_type)