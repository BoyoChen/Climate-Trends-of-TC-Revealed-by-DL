import numpy as np
import pandas as pd
import h5py
import os
import math
import pickle
from datetime import timedelta
from modules.image_processor import cart2polar
from modules.data_downloader import download_h5_data


def remove_outlier_and_nan(numpy_array, upper_bound=1000):
    numpy_array = np.nan_to_num(numpy_array, copy=False)
    numpy_array[numpy_array > upper_bound] = 0
    VIS = numpy_array[:, :, :, 2]
    VIS[VIS > 1] = 1  # VIS channel ranged from 0 to 1
    return numpy_array


def flip_SH_images(info_df, image_matrix):
    SH_idx = info_df.index[info_df.region == 'SH']
    image_matrix[SH_idx] = np.flip(image_matrix[SH_idx], 1)
    return image_matrix


def mark_good_quality_VIS(label_df, image_matrix):
    tmp_df = pd.DataFrame(columns=['vis_mean', 'vis_std'])
    for i in range(image_matrix.shape[0]):
        VIS_matrix = image_matrix[i, :, :, 2]
        tmp_df.loc[i] = [VIS_matrix.mean(), VIS_matrix.std()]

    tmp_df['hour'] = label_df.apply(lambda x: x.local_time.hour, axis=1)
    return tmp_df.apply(
        lambda x: (0.1 <= x.vis_mean <= 0.7) and (0.1 <= x.vis_std <= 0.31) and (7 <= x.hour <= 16),
        axis=1
    )


def fix_reversed_VIS(image_matrix):

    def scale_to_0_1(matrix):
        out = matrix - matrix.min()
        tmp_max = out.max()
        if tmp_max != 0:
            out /= tmp_max
        return out

    for i in range(image_matrix.shape[0]):
        IR1_matrix = image_matrix[i, :, :, 0]
        VIS_matrix = image_matrix[i, :, :, 2]
        reversed_VIS_matrix = 1 - VIS_matrix
        VIS_IR1_distance = abs(scale_to_0_1(IR1_matrix) - scale_to_0_1(VIS_matrix)).mean()
        reversed_VIS_IR1_distance = abs(scale_to_0_1(IR1_matrix) - scale_to_0_1(reversed_VIS_matrix)).mean()
        if reversed_VIS_IR1_distance > VIS_IR1_distance:
            VIS_matrix *= -1
            VIS_matrix += 1


def get_minutes_to_noon(local_time):
    minutes_in_day = 60 * local_time.hour + local_time.minute
    noon = 60 * 12
    return abs(noon - minutes_in_day)


def extract_label_and_feature_from_info(info_df):
    # --- region feature ---
    info_df['region_code'] = pd.Categorical(info_df.region).codes
    info_df['lon'] = (info_df.lon+180) % 360 - 180  # calibrate longitude, ex: 190 -> -170
    # --- time feature ---
    info_df['GMT_time'] = pd.to_datetime(info_df.time, format='%Y%m%d%H')
    info_df['local_time'] = info_df.GMT_time \
        + info_df.apply(lambda x: timedelta(hours=x.lon/15), axis=1)
    # --- year_day ---
    SH_idx = info_df.index[info_df.region == 'SH']
    info_df['yday'] = info_df.local_time.apply(lambda x: x.timetuple().tm_yday)
    info_df.loc[SH_idx, 'yday'] += 365 / 2  # TC from SH
    info_df['yday_transform'] = info_df.yday.apply(lambda x: x / 365 * 2 * math.pi)
    info_df['yday_sin'] = info_df.yday_transform.apply(lambda x: math.sin(x))
    info_df['yday_cos'] = info_df.yday_transform.apply(lambda x: math.cos(x))
    # --- hour ---
    info_df['hour_transform'] = info_df.apply(lambda x: x.local_time.hour / 24 * 2 * math.pi, axis=1)
    info_df['hour_sin'] = info_df.hour_transform.apply(lambda x: math.sin(x))
    info_df['hour_cos'] = info_df.hour_transform.apply(lambda x: math.cos(x))
    # --- add valid_profile column is not exisist ---
    info_df['valid_profile'] = False
    # split into 2 dataframe
    label_df = info_df[['region', 'ID', 'local_time', 'Vmax', 'R34', 'MSLP', 'valid_profile']]
    feature_df = info_df[['lon', 'lat', 'region_code', 'yday_cos', 'yday_sin', 'hour_cos', 'hour_sin']]
    return label_df, feature_df


def data_cleaning_and_organizing(images, info_df):
    label_df, feature_df = extract_label_and_feature_from_info(info_df)
    feature_df['minutes_to_noon'] = label_df['local_time'].apply(get_minutes_to_noon)

    # about the channel of "images" nparray, there is 2 cases: IR1+WV or IR1+WV+PMW+VIS
    # if PMW and VIS is missing, we need to append 2 fake channels.
    if images.shape[-1] == 2:
        images = np.concatenate([images, np.zeros_like(images)], axis=-1)
        feature_df['is_good_quality_VIS'] = False
    else:
        fix_reversed_VIS(images)
        feature_df['is_good_quality_VIS'] = mark_good_quality_VIS(label_df, images)

    images = remove_outlier_and_nan(images)
    images = flip_SH_images(info_df, images)
    return images, label_df, feature_df


def data_split(images, label_df, feature_df, structure_profiles, phase):
    if phase == 'train':
        target_index = label_df.index[label_df.ID < '2015000']
    elif phase == 'valid':
        target_index = label_df.index[np.logical_and('2017000' > label_df.ID, label_df.ID > '2015000')]
    elif phase == 'test':
        target_index = label_df.index[label_df.ID > '2017000']
    return {
        'label': label_df.loc[target_index].reset_index(drop=True),
        'feature': feature_df.loc[target_index].reset_index(drop=True),
        'image': images[target_index],
        'profile': structure_profiles[target_index]
    }


def extract_features_from_raw_file(data_folder, h5_name, coordinate):
    file_path = os.path.join(data_folder, h5_name)
    with h5py.File(file_path, 'r') as hf:
        images = hf['images'][:]
        if 'structure_profiles' in hf:
            structure_profiles = hf['structure_profiles'][:]
        else:
            structure_profiles = -np.ones([images.shape[0], 151])
    # collect info from every file in the list
    info_df = pd.read_hdf(file_path, key='info', mode='r')

    images, label_df, feature_df = data_cleaning_and_organizing(images, info_df)
    if coordinate == 'polar':
        images = np.array(list(map(cart2polar, images)))

    return images, label_df, feature_df, structure_profiles


def remove_bad_quality_VIS_data(label, feature, image, profile):
    good_VIS_index = feature.index[
        feature['is_good_quality_VIS']
    ]
    label = label.loc[good_VIS_index].reset_index(drop=True)
    feature = feature.loc[good_VIS_index].reset_index(drop=True)
    image = image[good_VIS_index]
    profile = profile[good_VIS_index]
    return label, feature, image, profile


def remove_invalid_profile_data(label, feature, image, profile):
    valid_profile_index = label.index[
        label['valid_profile'] > 0
    ]
    label = label.loc[valid_profile_index].reset_index(drop=True)
    feature = feature.loc[valid_profile_index].reset_index(drop=True)
    image = image[valid_profile_index]
    profile = profile[valid_profile_index]
    return label, feature, image, profile


def save_data_pickle_year_by_year(data_folder, coordinate, h5_name, pickle_path_format):
    images, label_df, feature_df, structure_profiles = extract_features_from_raw_file(data_folder, h5_name, coordinate)
    label_df['year'] = label_df.ID.map(lambda x: int(x[:4]))

    for y, year_df in label_df.groupby('year'):
        target_index = year_df.index
        year_dataset = {
            'label': year_df.drop(['year'], axis=1).reset_index(drop=True),
            'feature': feature_df.loc[target_index].reset_index(drop=True),
            'image': images[target_index],
            'profile': structure_profiles[target_index]
        }
        print(f'saving year {y} data pickle!')
        save_path = pickle_path_format % (data_folder, y, coordinate)
        with open(save_path, 'wb') as save_file:
            pickle.dump(year_dataset, save_file, protocol=5)


def load_dataset(data_folder, year_list, good_VIS_only=False, valid_profile_only=False, coordinate='cart'):
    year_datasets = []
    for year in year_list:
        print(f'now loading: year {year}')
        pickle_path_format = '%sTCSA.%s.%s.pickle'
        pickle_path = pickle_path_format % (data_folder, year, coordinate)

        if not os.path.isfile(pickle_path):
            print(f'pickle {pickle_path} not found! try to extract it from raw data!')
            download_h5_data(data_folder)
            h5_names = [file_name for file_name in os.listdir(data_folder) if file_name.endswith('.h5')]
            print(f'founded h5 files:\n{h5_names}')
            for h5_name in reversed(h5_names):
                print(f'processing {h5_name}')
                save_data_pickle_year_by_year(data_folder, coordinate, h5_name, pickle_path_format)

        with open(pickle_path, 'rb') as load_file:
            year_dataset = pickle.load(load_file)
        year_datasets.append(year_dataset)

    print('concat separated data')
    returned_dataset = {
        'label': pd.concat([year_dataset['label'] for year_dataset in year_datasets], ignore_index=True),
        'feature': pd.concat([year_dataset['feature'] for year_dataset in year_datasets], ignore_index=True),
        'image': np.concatenate([year_dataset['image'] for year_dataset in year_datasets], axis=0),
        'profile': np.concatenate([year_dataset['profile'] for year_dataset in year_datasets], axis=0)
    }

    if good_VIS_only or valid_profile_only:
        print('remove invalid data')
        if good_VIS_only:
            label, feature, image, profile = remove_bad_quality_VIS_data(**returned_dataset)
        if valid_profile_only:
            label, feature, image, profile = remove_invalid_profile_data(**returned_dataset)
        returned_dataset = {
            'label': label,
            'feature': feature,
            'image': image,
            'profile': profile
        }

    return returned_dataset
