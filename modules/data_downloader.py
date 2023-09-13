import os
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

file_dict = {
    'TCSA_2004_2018.h5': '18gQSmuGXEVDGDV9tsw00wSRhnWeNU4Xg'
}

def download_file(data_folder, file_name):
    file_path = os.path.join(data_folder, file_name)
    download_file_from_google_drive(file_dict[file_name], file_path)


def verify_data(data_folder):
    for file_name in file_dict:
        file_path = os.path.join(data_folder, file_name)
        if not os.path.isfile(file_path):
            print('data download failed!')
            return False
    return True


def download_h5_data(data_folder):
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)

    for file_name in file_dict:
        file_path = os.path.join(data_folder, file_name)
        if not os.path.isfile(file_path):
            download_file(data_folder, file_name)

    return verify_data(data_folder)
