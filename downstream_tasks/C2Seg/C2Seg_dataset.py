import torch
import torch.nn as nn
import torch.utils.data as Data
import slidingwindow as sw
import numpy as np
import scipy.io as sio
import os
import h5py
from sklearn.decomposition import PCA
from imgaug import augmenters as iaa

from torch.utils.data import Dataset


def getdata(dataset, patch, overlay, batchsize, pac_flag=False, band_norm_flag=False, aug_flag=False, distributed=False):
    label_train, label_valid, label_test, num_classes, band, total_length = slide_crop(dataset, patch, overlay, pac_flag,
                                                                             band_norm_flag, aug_flag)
    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(label_train)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(label_valid)
        test_sampler = torch.utils.data.distributed.DistributedSampler(label_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(label_train)
        valid_sampler = torch.utils.data.SequentialSampler(label_valid)
        test_sampler = torch.utils.data.SequentialSampler(label_test)

    label_test_loader = Data.DataLoader(label_test, batch_size=1, sampler=test_sampler, shuffle=False, drop_last=False)
    label_train_loader = Data.DataLoader(label_train, batch_size=batchsize, sampler=train_sampler, shuffle=False, num_workers=0,
                                         pin_memory=True, drop_last=True)
    label_valid_loader = Data.DataLoader(label_valid, batch_size=batchsize, sampler=valid_sampler, shuffle=False, num_workers=0,
                                         pin_memory=True, drop_last=True)
    return label_train_loader, label_valid_loader, label_test_loader, num_classes, band, total_length, train_sampler


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def band_normalization(data):
    """ normalize the matrix to (0,1), r.s.t A axis (Default=0)
        return normalized matrix and a record matrix for normalize back
    """
    size = data.shape
    if len(size) != 3:
        raise ValueError("Unknown dataset")
    for i in range(size[-1]):
        _range = np.max(data[:, :, i]) - np.min(data[:, :, i])
        data[:, :, i] = (data[:, :, i] - np.min(data[:, :, i])) / _range
    return data


def read_data(dataset, pca_flag=False, band_norm=False):
    if dataset == 'augsburg':
        num_classes, band = 14, 242
        train_file = r'/home/fengjq/5grade/seg/crosscity_data/data/data1/augsburg_multimodal.mat'
        col_train, row_train = 1360, 886
        valid_file = r'/home/fengjq/5grade/seg/crosscity_data/data/data1/berlin_multimodal.mat'
        col_valid, row_valid = 811, 2465
        input_data = sio.loadmat(train_file)
        valid_data = sio.loadmat(valid_file)

        hsi = input_data['HSI']  # 886 1360 242
        hsi = hsi.astype(np.float32)
        msi = input_data['MSI']
        msi = msi.astype(np.float32)
        sar = input_data['SAR']
        sar = sar.astype(np.float32)
        label = input_data['label']
        

        hsi_valid = valid_data['HSI'][:, :, 0:band]  # 2456 811 242
        hsi_valid = hsi_valid.astype(np.float32)
        msi_valid = valid_data['MSI']
        msi_valid = msi_valid.astype(np.float32)
        sar_valid = valid_data['SAR']
        sar_valid = sar_valid.astype(np.float32)
        label_valid = valid_data['label']

        # 存高光谱图像为tiff
        print(1)
        import tifffile as tiff
        normalized_hsi = np.clip(hsi, 0, 65535).astype(np.uint16)
        metadata = {
            'Description': 'HSI data with 242 channels',  # Custom header or description
            'Channels': 242,
            'Height': hsi.shape[0],
            'Width': hsi.shape[1],
            'Data type': 'uint16',
        }
        tiff.imwrite("output_image.tiff", normalized_hsi, description=metadata['Description'], metadata=metadata)
        


        # a = np.min(label_valid)
        # PCA
        if pca_flag:
            hsi_matrix = np.reshape(hsi, (hsi.shape[0] * hsi.shape[1], hsi.shape[2]))  # 2456*811 242
            pca = PCA(n_components=72)
            pca.fit_transform(hsi_matrix)
            newspace = pca.components_
            newspace = newspace.transpose()  # 242*72
            hsi_matrix = np.matmul(hsi_matrix, newspace)  # 2456*811 72
            hsi = np.reshape(hsi_matrix, (hsi.shape[0], hsi.shape[1], pca.n_components_))

            hsi_matrix = np.reshape(hsi_valid,
                                    (hsi_valid.shape[0] * hsi_valid.shape[1], hsi_valid.shape[2]))  # 2456*811 242
            # 采用相同的映射方法，避免训练集和验证集空间不同
            # pca = PCA(n_components=72)
            # pca.fit_transform(hsi_matrix)
            # newspace = pca.components_
            # newspace = newspace.transpose()  # 242*72
            hsi_matrix = np.matmul(hsi_matrix, newspace)  # 2456*811 72
            hsi_valid = np.reshape(hsi_matrix, (hsi_valid.shape[0], hsi_valid.shape[1], pca.n_components_))

            band = 72

            del hsi_matrix
        else:
            # 删除噪声波段,同时保证数据满足3的倍数
            invalid_channels = [126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 160, 161, 162, 163, 164, 165, 166]
            valid_channels_ids = [c+1 for c in range(224) if c not in invalid_channels]
            # 202-1
            valid_channels_ids = valid_channels_ids[:-1]
            hsi = hsi[:,:,valid_channels_ids]
            hsi_valid = hsi_valid[:,:,valid_channels_ids]
            band = 201
            

    elif dataset == 'beijing':
        num_classes = 14
        band = 10  # 116
        ## beijing is training, wuhan is testing
        train_file = r'/home/fengjq/5grade/seg/crosscity_data/data/data2/beijing.mat'
        train_file_label = r'/home/fengjq/5grade/seg/crosscity_data/data/data2/beijing_label.mat'
        col_train, row_train = 13474, 8706
        valid_file = r'/home/fengjq/5grade/seg/crosscity_data/data/data2/wuhan.mat'
        valid_file_label = r'/home/fengjq/5grade/seg/crosscity_data/data/data2/wuhan_label.mat'
        col_valid, row_valid = 6225, 8670

        with h5py.File(train_file, 'r') as f:
            f = h5py.File(train_file, 'r')
        hsi = np.array(f['HSI'])
        msi = np.transpose(f['MSI'])
        sar = np.transpose(f['SAR'])
        # idx = np.where(np.isnan(sar))
        sar[1097, 8105, 1] = sum([sar[1096, 8105, 1], sar[1098, 8105, 1], sar[1097, 8104, 1], sar[1097, 8106, 1]]) / 4

        # cut beijing suburb region
        cut_length = 0
        col_train = col_train - cut_length
        hsi = hsi[:, :, cut_length // 3:]
        msi = msi[cut_length:, :, :]
        sar = sar[cut_length:, :, :]

        if pca_flag:
            # applying PCA for HSI # hsi (116, 2903, 4492)
            hsi_matrix = np.reshape(np.transpose(hsi), (hsi.shape[1] * hsi.shape[2], hsi.shape[0]))  # 2903*4492 116
            pca = PCA(n_components=72)
            pca.fit_transform(hsi_matrix)
            newspace = pca.components_
            newspace = newspace.transpose()  # 116*72
            hsi_matrix = np.matmul(hsi_matrix, newspace)  # 2903*4492 72
            hsi_cube = np.transpose(np.reshape(hsi_matrix, (hsi.shape[2], hsi.shape[1], pca.n_components_)))
            band = 72
            del hsi
        else:
            # 116+1
            mean_matrix_expanded = np.expand_dims(np.mean(hsi, axis=0), axis=0)
            hsi_cube = np.concatenate((hsi, mean_matrix_expanded), axis=0)
            band = 117

        mm = nn.Upsample(scale_factor=3, mode='nearest', align_corners=None)
        # upsample from 30m to 10m
        hsi1 = mm(torch.from_numpy(hsi_cube).unsqueeze(0)).squeeze().numpy()
        hsi1 = np.transpose(hsi1)
        # remove extra pixels
        hsi = hsi1[:col_train, :row_train, :]
        del hsi1

        with h5py.File(train_file_label, 'r') as f:
            f = h5py.File(train_file_label, 'r')
        label = np.transpose(f['label'])
        # cut beijing label
        label = label[cut_length:, :]

        with h5py.File(valid_file, 'r') as f:
            f = h5py.File(valid_file, 'r')
        hsi_valid = np.array(f['HSI'])
        msi_valid = np.transpose(f['MSI'])
        sar_valid = np.transpose(f['SAR'])

        if pca_flag:
            ## applying PCA for valid HSI
            hsi_matrix = np.reshape(np.transpose(hsi_valid), (hsi_valid.shape[1] * hsi_valid.shape[2], hsi_valid.shape[0]))
            # 采用相同的映射方法，避免训练集和验证集空间不同
            # pca = PCA(n_components=72)
            # pca.fit_transform(hsi_matrix)
            # newspace = pca.components_
            # newspace = newspace.transpose()
            hsi_matrix = np.matmul(hsi_matrix, newspace)
            hsi_cube = np.transpose(np.reshape(hsi_matrix, (hsi_valid.shape[2], hsi_valid.shape[1], pca.n_components_)))
            del hsi_valid
        else:
            # 116+1
            mean_matrix_expanded = np.expand_dims(np.mean(hsi_valid, axis=0), axis=0)
            hsi_cube = np.concatenate((hsi_valid, mean_matrix_expanded), axis=0)

        hsi1 = mm(torch.from_numpy(hsi_cube).unsqueeze(0)).squeeze().numpy()
        hsi_valid = np.transpose(hsi1)
        del hsi1

        with h5py.File(valid_file_label, 'r') as f:
            f = h5py.File(valid_file_label, 'r')
        label_valid = np.transpose(f['label'])

    else:
        raise ValueError("Unknown dataset")

    # normalize data
    if band_norm:
        norm = band_normalization
    else:
        norm = normalization

    hsi = norm(hsi)
    msi = norm(msi)
    sar = norm(sar)
    hsi_valid = norm(hsi_valid)
    msi_valid = norm(msi_valid)
    sar_valid = norm(sar_valid)

    return hsi, msi, sar, label, hsi_valid, msi_valid, sar_valid, label_valid, num_classes, band




def slide_crop(dataset, patch, overlay, pca_flag=False, band_norm_flag=False, aug_flag=False):
    hsi, msi, sar, label, hsi_valid, msi_valid, sar_valid, label_valid, num_classes, band = read_data(dataset, pca_flag,
                                                                                                      band_norm_flag)

    if dataset == 'augsburg':
        col_train, row_train = 1360, 886
        col_valid, row_valid = 811, 2465

    elif dataset == 'beijing':
        col_train, row_train = 8706, 13474
        col_valid, row_valid = 8670, 6225

    else:
        raise ValueError("Unknown dataset")

    transform = iaa.Sequential([
        iaa.Rot90([0, 1, 2, 3]),
        iaa.VerticalFlip(p=0.5),
        iaa.HorizontalFlip(p=0.5),
    ])

    # slide crop for train data
    window_set_train = sw.generate(hsi, sw.DimOrder.HeightWidthChannel, patch, overlay)
    hsi_list = []
    msi_list = []
    sar_list = []
    label_list = []
    id_list = []
    color_dic = {
        0: "#000000", 1: '#00FFFF', 2: '#FFFFFF', 3: '#FF0000', 4: '#DDA0DD',
        5: '#9400D3', 6: '#FF00FF', 7: '#FFFF00', 8: '#CD853F', 9: '#BDB76B',
        10: '#00FF00', 11: '#9ACD32', 12: '#8B4513', 13: '#483D8B'
    }
    # 将十六进制颜色转换为BGR格式
    def hex_to_bgr(hex_color):
        hex_color = hex_color.lstrip('#')
        rgb = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
        return (rgb[2], rgb[1], rgb[0])  # OpenCV uses BGR format

    for window in window_set_train:
        subset_hsi = hsi[window.indices()]
        subset_msi = msi[window.indices()]
        subset_sar = sar[window.indices()]
        subset_label = label[window.indices()]
        if aug_flag:
            all_img = np.concatenate((subset_hsi, subset_msi, subset_sar), axis=-1)
            img1, label1 = transform(image=all_img,
                                     segmentation_maps=np.stack(
                                         (subset_label[np.newaxis, :, :], subset_label[np.newaxis, :, :])
                                         , axis=-1).astype(np.int32))

            subset_label = label1[0, :, :, 0]
            subset_hsi = img1[:, :, :band]
            subset_msi = img1[:, :, band:band+4]
            subset_sar = img1[:, :, band+4:band+6]

        hsi_list.append(subset_hsi)
        msi_list.append(subset_msi)
        sar_list.append(subset_sar)
        label_list.append(subset_label)
        id_list.append(str(window.x)+'_'+str(window.y))
        # # # 保存图像：
        # import cv2
        # root_dir = '/dev1/fengjq/IEEE_TPAMI_SpectralGPT/multi_train/C2Seg/20250509_new_vis_test/'
        # msi_name = dataset + '/msi/'+ 'train_img_' + str(window.x) + '_'  + str(window.y) + '.png'
        # label_name = dataset + '/label/' + 'train_gt_' + str(window.x) + '_'  + str(window.y) + '.png'
        # msi_rgb = np.clip(np.sqrt(subset_msi[:,:,:3]) * 255, 0, 255).astype(np.uint8)
        # # 如果目录不存在则创建
        # dir_path = os.path.dirname(root_dir+msi_name)
        # os.makedirs(dir_path, exist_ok=True)
        # cv2.imwrite(root_dir+msi_name, msi_rgb)
        # # 创建一个空的 RGB 图像（高度 x 宽度 x 3）
        # height, width = subset_label.shape
        # label_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        # # 根据 color_dic 填充图像
        # for _label, color in color_dic.items():
        #     bgr_color = hex_to_bgr(color)
        #     label_rgb[subset_label == _label] = bgr_color  
        # # 如果目录不存在则创建
        # dir_path = os.path.dirname(root_dir+label_name)
        # os.makedirs(dir_path, exist_ok=True)
        # cv2.imwrite(root_dir+label_name, label_rgb)

    del hsi, msi, sar, label
    hsi_list = np.array(hsi_list)
    msi_list = np.array(msi_list)
    sar_list = np.array(sar_list)
    label_list = np.array(label_list)
    id_list = np.array(id_list)
    # has_zero = np.any(label_list==0)
    # print(has_zero)

    # non-overlay crop for valid data
    window_set_valid = sw.generate(hsi_valid, sw.DimOrder.HeightWidthChannel, patch, overlay)
    hsi_valid_list = []
    msi_valid_list = []
    sar_valid_list = []
    label_valid_list = []
    id_valid_list = []
    for window in window_set_valid:
        subset_hsi = hsi_valid[window.indices()]
        subset_msi = msi_valid[window.indices()]
        subset_sar = sar_valid[window.indices()]
        subset_label = label_valid[window.indices()]
        if aug_flag:
            all_img = np.concatenate((subset_hsi, subset_msi, subset_sar), axis=-1)
            img1, label1 = transform(image=all_img,
                                     segmentation_maps=np.stack(
                                         (subset_label[np.newaxis, :, :], subset_label[np.newaxis, :, :])
                                         , axis=-1).astype(np.int32))

            subset_label = label1[0, :, :, 0]
            subset_hsi = img1[:, :, :band]
            subset_msi = img1[:, :, band:band+4]
            subset_sar = img1[:, :, band+4:band+6]
        hsi_valid_list.append(subset_hsi)
        msi_valid_list.append(subset_msi)
        sar_valid_list.append(subset_sar)
        label_valid_list.append(subset_label)
        id_valid_list.append(str(window.x)+'_'+str(window.y))
 
    hsi_valid_list = np.array(hsi_valid_list)
    msi_valid_list = np.array(msi_valid_list)
    sar_valid_list = np.array(sar_valid_list)
    label_valid_list = np.array(label_valid_list)
    id_valid_list =  np.array(id_valid_list)

    # non-overlay crop for valid data
    window_set_test = sw.generate(hsi_valid, sw.DimOrder.HeightWidthChannel, patch, 0)
    hsi_test_list = []
    msi_test_list = []
    sar_test_list = []
    label_test_list = []
    id_test_list = []
    for window in window_set_test:
        subset_hsi = hsi_valid[window.indices()]
        subset_msi = msi_valid[window.indices()]
        subset_sar = sar_valid[window.indices()]
        subset_label = label_valid[window.indices()]
        hsi_test_list.append(subset_hsi)
        msi_test_list.append(subset_msi)
        sar_test_list.append(subset_sar)
        label_test_list.append(subset_label)
        id_test_list.append(str(window.x)+'_'+str(window.y))
        # # 保存图像：
        # import cv2
        # root_dir = '/home/fengjq/5grade/seg/crosscity_data/vis/'
        # dataset_test = 'berlin' if dataset == 'augsburg' else 'wuhan'
        # msi_name = dataset_test + '/msi/'+ 'train_img_' + str(window.x) + '_'  + str(window.y) + '.png'
        # label_name = dataset_test + '/label/' + 'train_gt_' + str(window.x) + '_'  + str(window.y) + '.png'
        # msi_rgb = np.clip(np.sqrt(subset_msi[:,:,:3]) * 255, 0, 255).astype(np.uint8)
        # cv2.imwrite(root_dir+msi_name, msi_rgb)
        # # 创建一个空的 RGB 图像（高度 x 宽度 x 3）
        # height, width = subset_label.shape
        # label_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        # # 根据 color_dic 填充图像
        # for _label, color in color_dic.items():
        #     bgr_color = hex_to_bgr(color)
        #     label_rgb[subset_label == _label] = bgr_color  
        # cv2.imwrite(root_dir+label_name, label_rgb)

    del hsi_valid, msi_valid, sar_valid, label_valid
    hsi_test_list = np.array(hsi_test_list)
    msi_test_list = np.array(msi_test_list)
    sar_test_list = np.array(sar_test_list)
    label_test_list = np.array(label_test_list)
    id_test_list = np.array(id_test_list)

    # construct dataset
    hsi_list = torch.from_numpy(hsi_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    msi_list = torch.from_numpy(msi_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    sar_list = torch.from_numpy(sar_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    label_list = torch.from_numpy(label_list).type(torch.LongTensor)
    # label_train = Data.TensorDataset(hsi_list, msi_list, sar_list, label_list)
    label_train = C2SegDataset(hsi_list, msi_list, sar_list, label_list, id_list)

    hsi_valid_list = torch.from_numpy(hsi_valid_list[:, :, :, :band].transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    msi_valid_list = torch.from_numpy(msi_valid_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    sar_valid_list = torch.from_numpy(sar_valid_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    label_valid_list = torch.from_numpy(label_valid_list).type(torch.LongTensor)
    # label_valid = Data.TensorDataset(hsi_valid_list, msi_valid_list, sar_valid_list, label_valid_list)
    label_valid = C2SegDataset(hsi_valid_list, msi_valid_list, sar_valid_list, label_valid_list, id_valid_list)

    hsi_test_list = torch.from_numpy(hsi_test_list[:, :, :, :band].transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    msi_test_list = torch.from_numpy(msi_test_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    sar_test_list = torch.from_numpy(sar_test_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    label_test_list = torch.from_numpy(label_test_list).type(torch.LongTensor)
    # label_test = Data.TensorDataset(hsi_test_list, msi_test_list, sar_test_list, label_test_list)
    label_test = C2SegDataset(hsi_test_list, msi_test_list, sar_test_list, label_test_list, id_test_list)

    return label_train, label_valid, label_test, num_classes, band, len(label_list)


def slide_crop_all_modalities(dataset, patch, overlay):
    hsi, msi, sar, label, hsi_valid, msi_valid, sar_valid, label_valid, num_classes, band = read_data(dataset)

    if dataset == 'augsburg':
        col_train, row_train = 1360, 886
        col_valid, row_valid = 811, 2465

    elif dataset == 'beijing':
        col_train, row_train = 8706, 13474
        col_valid, row_valid = 8670, 6225

    else:
        raise ValueError("Unknown dataset")

    # slide crop for train data
    img = np.concatenate([hsi, msi, sar], axis=2)  # w, h, c
    del hsi, msi, sar
    window_set_train = sw.generate(img, sw.DimOrder.HeightWidthChannel, patch, overlay)
    img_list = []
    label_list = []
    for window in window_set_train:
        subset_img = img[window.indices()]
        subset_label = label[window.indices()]
        img_list.append(subset_img)
        label_list.append(subset_label)
    img_list = np.array(img_list)
    label_list = np.array(label_list)
    del img, label

    # non-overlay crop for valid data
    img_valid = np.concatenate([hsi_valid, msi_valid, sar_valid], axis=2)
    del hsi_valid, msi_valid, sar_valid

    window_set_valid = sw.generate(img_valid, sw.DimOrder.HeightWidthChannel, patch, 0)
    img_valid_list = []
    label_valid_list = []
    for window in window_set_valid:
        subset_img = img_valid[window.indices()]
        subset_label = label_valid[window.indices()]
        img_valid_list.append(subset_img)
        label_valid_list.append(subset_label)
    img_valid_list = np.array(img_valid_list)
    label_valid_list = np.array(label_valid_list)
    del img_valid, label_valid

    # construct dataset
    img_list = torch.from_numpy(img_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    label_list = torch.from_numpy(label_list).type(torch.LongTensor)
    label_train = Data.TensorDataset(img_list, label_list)

    img_valid_list = torch.from_numpy(img_valid_list.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    label_valid_list = torch.from_numpy(label_valid_list).type(torch.LongTensor)
    label_valid = Data.TensorDataset(img_valid_list, label_valid_list)

    return label_train, label_valid, num_classes, band

class C2SegDataset(Dataset):
    def __init__(self, hsi_list, msi_list, sar_list, label_list, id_list=None):
        super(C2SegDataset, self).__init__()
        self.hsi_list = hsi_list
        self.msi_list = msi_list
        self.sar_list = sar_list
        self.label_list = label_list
        self.id_list = id_list

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        hsi = self.hsi_list[index]
        msi = self.msi_list[index]
        sar = self.sar_list[index]
        label = self.label_list[index]
        if self.id_list is not None:
            return hsi, label, self.id_list[index]
    
        return hsi, label

    def __len__(self):
        return len(self.label_list)