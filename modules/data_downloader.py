import wget
import tarfile
import os
import ssl


file_dict = {
    'TCSA.h5': 'https://learner.csie.ntu.edu.tw/~boyochen/TCSA/TCSA.h5.tar.gz'
}

compressed_postfix = '.tar.gz'


def download_compressed_file(data_folder, file_name):
    ssl._create_default_https_context = ssl._create_unverified_context
    file_url = file_dict[file_name]
    file_path = os.path.join(data_folder, file_name + compressed_postfix)
    wget.download(file_url, out=file_path)


def uncompress_file(data_folder, file_name):
    compressed_file_path = os.path.join(data_folder, file_name + compressed_postfix)
    if not os.path.isfile(compressed_file_path):
        download_compressed_file(data_folder, file_name)

    with tarfile.open(compressed_file_path) as tar:
        tar.extractall(path=data_folder)


def verify_data(data_folder):
    for file_name in file_dict:
        file_path = os.path.join(data_folder, file_name)
        if not os.path.isfile(file_path):
            print('data download failed!')
            return False
    return True


def download_data(data_folder, h5_name):
    # !!!!!!!!!!!!!!!!!!!!!!!!!!
    # need rewrite.
    # given h5 file name, download the exactly one to data_folder.
    # !!!!!!!!!!!!!!!!!!!!!!!!!!
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)

    for file_name in file_dict:
        file_path = os.path.join(data_folder, file_name)
        if not os.path.isfile(file_path):
            uncompress_file(data_folder, file_name)

    return verify_data(data_folder)
