import os
# import tarfile
from PIL import Image
# import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
CLASS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile',
    'toothbrush', 'transistor', 'wood', 'zipper'
]


class MVTecDataset(Dataset):
    def __init__(self,
                 dataset_path=os.path.join('data','mvtec_anomaly_detection'),
                 class_name='transistor',
                 is_train=True,
                 resize=256,
                 ):
        print(class_name)
        #assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        print(self.dataset_path)
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        # self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        # download dataset if not exist
        # self.download()

        # load dataset
        self.recover,self.recover_1,self.recover_2,self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        self.transform_x = transforms.Compose([
            transforms.Resize(resize, Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_mask = transforms.Compose(
            [transforms.Resize(resize, Image.NEAREST),
             transforms.ToTensor()])

    def __getitem__(self, idx):
        recover,recover_1,recover_2,x, y, mask =self.recover[idx], self.recover_1[idx],self.recover_2[idx],self.x[idx], self.y[idx], self.mask[idx]
        recover = Image.open(recover).convert('RGB')
        recover = self.transform_x(recover)

        recover_1 = Image.open(recover_1).convert('RGB')
        recover_1 = self.transform_x(recover_1)

        recover_2 = Image.open(recover_2).convert('RGB')
        recover_2 = self.transform_x(recover_2)

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.resize, self.resize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
     

        return recover,recover_1,recover_2,x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        recover,recover_1,recover_2,x, y, mask,ori = [],[],[],[], [], [],[]

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')
        recover_dir = os.path.join('mvtec_anomaly_detection_8', self.class_name, 'test')
        print(recover_dir)
        #recover_1_dir = os.path.join('mvtec_anomaly_detection_1', self.class_name, 'recover')
        #recover_2_dir = os.path.join('mvtec_anomaly_detection_1', self.class_name, 'recover')
        img_types = sorted(os.listdir(img_dir))
        recover_types = sorted(os.listdir(recover_dir))
        #recover_1_types = sorted(os.listdir(recover_1_dir))
        #recover_2_types = sorted(os.listdir(recover_2_dir))
        for recover_type in recover_types:
            recover_type_dir = os.path.join(recover_dir, recover_type)
            if not os.path.isdir(recover_type_dir):
                continue
            recover_fpath_list = sorted(
                [os.path.join(recover_type_dir, f) for f in os.listdir(recover_type_dir) if f.endswith('.png')])
            recover.extend(recover_fpath_list)
            ori_fpath_list = sorted(
                [os.path.join(recover_type_dir, f) for f in os.listdir(recover_type_dir) if f.endswith('ori.jpg')])
            ori.extend(ori_fpath_list)
        #for recover_type in recover_1_types:
            #recover_type_dir = os.path.join(recover_1_dir, recover_type)
            #if not os.path.isdir(recover_type_dir):
                #continue
            #recover_fpath_list = sorted(
                #[os.path.join(recover_type_dir, f) for f in os.listdir(recover_type_dir) if f.endswith('.jpg')])
            #recover_1.extend(recover_fpath_list)
        #for recover_type in recover_2_types:
            #recover_type_dir = os.path.join(recover_2_dir, recover_type)
            #if not os.path.isdir(recover_type_dir):
                #continue
           #recover_fpath_list = sorted(
                #[os.path.join(recover_type_dir, f) for f in os.listdir(recover_type_dir) if f.endswith('.jpg')])
            #recover_2.extend(recover_fpath_list)
        #for img_type in img_types:

            # load images
            #img_type_dir = os.path.join(img_dir, img_type)
            #if not os.path.isdir(img_type_dir):
                #continue
            #img_fpath_list = sorted(
                #[os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.jpg')])
            #x.extend(img_fpath_list)
        for i in recover:
            x.append(i.replace('mvtec_anomaly_detection_8','mvtec_anomaly_detection'))
            #recover_1.append(i.replace('mvtec_anomaly_detection_1','mvtec_anomaly_detection_2'))
            #recover_2.append(i.replace('mvtec_anomaly_detection_1','mvtec_anomaly_detection_3'))
            # load gt labels
        #print(x)
        for j in x:
            if j.split(r'/')[3] == 'good':
                #y.extend([0] * len(img_fpath_list))
                #mask.extend([None] * len(img_fpath_list))
                y.append(0)
                mask.append(None)
            else:
                y.append(1)
                #gt_type_dir = os.path.join(gt_dir, img_type)
                #img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                #gt_fpath_list = [os.path.join(gt_type_dir, mask_true) for mask_true in os.listdir(gt_type_dir) if mask_true.endswith('.png')]
                #mask.extend(gt_fpath_list)
                mask.append(j.replace('test','ground_truth').replace('.png','_mask.png'))
        #print(len(list(mask)))
        #print(len(list(recover)))
        #for i in range(len(list(recover))):
            #print('recover')
            #print(list(recover)[i])
            #print('img')
            #print(list(x)[i])
            #print('mask')
            #print(list(mask)[i])
        assert len(x) == len(y), 'number of x and y should be same'
        assert len(x) == len(recover), 'number of x and recover should be same'
        assert len(recover_1) == len(recover_2), 'number of recover and recover should be same'
        return list(recover),list(recover),list(recover),list(x), list(y), list(mask)


#     def download(self):
#         """Download dataset if not exist"""

#         if not os.path.exists(self.mvtec_folder_path):
#             tar_file_path = self.mvtec_folder_path + '.tar.xz'
#             if not os.path.exists(tar_file_path):
#                 download_url(URL, tar_file_path)
#             print('unzip downloaded dataset: %s' % tar_file_path)
#             tar = tarfile.open(tar_file_path, 'r:xz')
#             tar.extractall(self.mvtec_folder_path)
#             tar.close()

#         return

# class DownloadProgressBar(tqdm):
#     def update_to(self, b=1, bsize=1, tsize=None):
#         if tsize is not None:
#             self.total = tsize
#         self.update(b * bsize - self.n)

# def download_url(url, output_path):
#     with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
#         urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
