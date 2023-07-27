import logging
import pickle
import os
from re import S

import numpy as np
from torchvision import transforms
from learn2learn.vision.datasets import MiniImagenet, FullOmniglot

from face_auditor.lib_dataset.meta_dataset import MetaDataset, UnionMetaDataset, FilteredMetaDataset, SplitMetaDataset
from face_auditor.lib_dataset.datasets.vggface2 import VGGFace2
from face_auditor.lib_dataset.datasets.webface import WebFace
from face_auditor.lib_dataset.datasets.umdfaces import UmdFaces
from face_auditor.lib_dataset.datasets.celeba import CelebA
from face_auditor.parameter_parser import parameter_parser
import face_auditor.config as config


class Preprocess:
    def __init__(self, args, dataset_name):
        self.logger = logging.getLogger('preprocess')

        self.dataset_name = dataset_name
        self.args = args
        self.raw_data_path = os.path.join(config.RAW_DATA_PATH, str(dataset_name))
        self.image_size = (self.args['image_size'], self.args['image_size'])  # 28*28, 32*32, 224*224, 96*96
        self.transforms = transforms.Compose([
            transforms.RandomAffine(15),
            transforms.ToTensor()
        ])
        self.transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.topn = None
        self.n_images = None

    def load_data(self):
        self.logger.info('loading data %s' % self.dataset_name)

        if self.dataset_name == 'webface':
            self.dataset, self.topn, self.n_images = self._load_webface()
        elif self.dataset_name == 'vggface2':
            self.dataset, self.topn, self.n_images = self._load_vggface2()
        elif self.dataset_name == 'umdfaces':
            self.dataset, self.topn, self.n_images = self._load_umdfaces()
        elif self.dataset_name == 'celeba':
            self.dataset, self.topn, self.n_images = self._load_celeba()
        else:
            raise Exception('unsupported dataset')
        self.logger.info('loading data finished')

    def split_data(self):
        self.logger.info('splitting data')

        victim_ratio = self.args['victim_ratio']
        test_ratio = self.args['test_ratio']
        assert victim_ratio >= 0 and victim_ratio <= 1, "Victim data ratio must be in [0, 1]"
        assert test_ratio > 0 and test_ratio < 1, "Test data ratio must be in (0, 1)"

        victim_train = victim_ratio * (1 - test_ratio)
        victim_test = victim_ratio * test_ratio
        adv_train = (1 - victim_ratio) * (1 - test_ratio)
        adv_test = (1 - victim_ratio) * test_ratio

        # Original paper has 50:50 victim:adv split
        # Add support for victim_ratio=0/1 to allow different ratios of data to be used in
        # Cross-dataset experiments
        shadow_train_indices, shadow_test_indices = None, None
        target_train_indices, target_test_indices = None, None
        if victim_ratio > 0 and victim_ratio < 1:
            target_train_indices, target_test_indices, shadow_train_indices, shadow_test_indices = \
                self._split_indices(len(self.dataset.labels),
                                    (victim_train, victim_test, adv_train, adv_test))
        elif victim_ratio == 0:
            shadow_train_indices, shadow_test_indices = self._split_indices(len(self.dataset.labels), (1 - test_ratio, test_ratio))
        else:
            target_train_indices, target_test_indices = self._split_indices(len(self.dataset.labels), (1 - test_ratio, test_ratio))

        # The 0.5 here comes form split for members and non-member images to be used per person
        total_subjects_audit = int(self.n_images)
        train_mem_indices, train_nonmem_indices = self._split_indices(total_subjects_audit, (0.5, 0.5))  # for celeba

        self.target_train_dset, self.target_test_dset = None, None
        self.shadow_train_dset, self.shadow_test_dset = None, None
        self.target_train_mem_dset, self.target_train_nonmem_dset = None, None
        self.shadow_train_mem_dset, self.shadow_train_nonmem_dset = None, None
        
        # Any target dataa present
        if target_train_indices is not None:
            self.target_train_dset = FilteredMetaDataset(self.dataset, target_train_indices)
            self.target_test_dset = FilteredMetaDataset(self.dataset, target_test_indices)
            self.target_train_mem_dset = SplitMetaDataset(self.target_train_dset, train_mem_indices)
            self.target_train_nonmem_dset = SplitMetaDataset(self.target_train_dset, train_nonmem_indices)

        # Any shadow data present
        if shadow_train_indices is not None:
            self.shadow_train_dset = FilteredMetaDataset(self.dataset, shadow_train_indices)
            self.shadow_test_dset = FilteredMetaDataset(self.dataset, shadow_test_indices)
            self.shadow_train_mem_dset = SplitMetaDataset(self.shadow_train_dset, train_mem_indices)
            self.shadow_train_nonmem_dset = SplitMetaDataset(self.shadow_train_dset, train_nonmem_indices)


    def save_data(self):
        self.logger.info('saving data')
        data_path = config.PROCESSED_DATA_PATH + str(self.args['image_size']) + "/" + "_".join((str(self.args['dataset_task']), self.dataset_name))
        # Add ratio-related information to split as well
        data_path += "_%.2f_%.2f" % (self.args['victim_ratio'], self.args['test_ratio'])
        if self.args['is_adv_defense']:
            data_path = data_path + "_" + self.args['fawkes_mode']
        print(data_path)

        pickle.dump((self.target_train_dset, self.shadow_train_dset,
                     self.target_train_mem_dset, self.target_train_nonmem_dset,
                     self.shadow_train_mem_dset, self.shadow_train_nonmem_dset,
                     self.target_test_dset, self.shadow_test_dset),
                    open(data_path, 'wb'), protocol=4)

    def _split_indices(self, num_labels: int, split_ratio: float):
        self.split_indices = []
        remain_indices = np.arange(num_labels)

        for ratio in split_ratio[:-1]:
            num_samples = int(num_labels * ratio)
            sample_indices = np.random.choice(remain_indices, num_samples, replace=False)
            self.split_indices.append(sample_indices)
            remain_indices = np.setdiff1d(remain_indices, sample_indices)

        self.split_indices.append(remain_indices)
        return tuple(self.split_indices)

    def _load_miniimagenet(self):
        transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize(self.image_size),
            # transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dataset = MiniImagenet(root=self.raw_data_path, transform=transform, mode='train', download=False)
        valid_dataset = MiniImagenet(root=self.raw_data_path, transform=transform, mode='validation', download=False)
        test_dataset = MiniImagenet(root=self.raw_data_path, transform=transform, mode='test', download=False)

        return UnionMetaDataset([train_dataset, valid_dataset, test_dataset])

    def _load_full_omniglot(self):
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            lambda x: 1.0 - x,
        ])
        dataset = FullOmniglot(root=self.raw_data_path, transform=transform, download=True)
        return MetaDataset(dataset)

    def _load_webface(self):
        TOPN={
            0: 400, # 162+
            1: 888, # 96+
            2: 1629, # 64+
            3: 4032, # 32+
            4: 6864, # 20+
            5: 827, # 100+
        }
        NImages={
            0: 32, # 87+
            1: 96, # 96+
            2: 64, # 64+
            3: 32, # 32+
            4: 20, # 20+
            5: 100, # 100+
        }
        # topn means we only use the topn users.
        topn=TOPN[self.args['dataset_task']]
        n_images=NImages[self.args['dataset_task']]
        dataset = WebFace(root=self.raw_data_path, transform=self.transform, topn=topn, num_imges=n_images)
        return MetaDataset(dataset), topn, n_images

    def _load_umdfaces(self):
        TOPN={
            0: 300, # 89+
            1: 232, # 96+
            2: 1246, # 64+
            3: 5885, # 32+
            4: 7173, # 20+
            5: 200, # 100+
        }
        NImages={
            0: 100, # 87+
            1: 96, # 96+
            2: 64, # 64+
            3: 32, # 32+
            4: 20, # 20+
            5: 100, # 100+
        }
        topn=TOPN[self.args['dataset_task']]
        n_images=NImages[self.args['dataset_task']]
        dataset = UmdFaces(root=self.raw_data_path + 'images/images_low/', transform=self.transform, topn=topn, num_imges=n_images)
        return MetaDataset(dataset), topn, n_images

    def _load_vggface2(self):
        TOPN={
            0: 5260, # 87+
            1: 5258, # 96+
            2: 5260, # 64+
            3: 5260, # 32+
            4: 5260, # 20+
            5: 5257, # 100+
        }
        NImages={
            0: 32, # 87+
            1: 96, # 96+
            2: 64, # 64+
            3: 32, # 32+
            4: 20, # 20+
            5: 100, # 100+
        }
        topn=TOPN[self.args['dataset_task']]
        n_images=NImages[self.args['dataset_task']]
        dataset = VGGFace2(root=self.raw_data_path, transform=self.transform, topn=topn, num_imges=n_images)
        return MetaDataset(dataset), topn, n_images

    def _load_celeba(self):
        TOPN={
            0: 6348, # 20+
            1: 0, # 96+
            2: 0, # 64+
            3: 8, # 32+
            4: 6348, # 20+
            5: 6347, # 20+
        }
        NImages={
            0: 20, # 20+
            1: 20, # 96+
            2: 20, # 64+
            3: 20, # 32+
            4: 20, # 20+
            5: 20, # 20+
        }
        topn=TOPN[self.args['dataset_task']]
        n_images=NImages[self.args['dataset_task']]
        dataset = CelebA(root=self.raw_data_path, transform=self.transform, topn=topn, num_imges=n_images, mode=self.args['fawkes_mode'])
        return MetaDataset(dataset), topn, n_images


if __name__ == '__main__':
    os.chdir('../')

    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)
    args = parameter_parser()
    # dataset_name_list = ['miniimagenet', 'full_omniglot', 'fc100', 'lfw', 'webface', 'vggface2', 'umdfaces', 'celeba']
    # dataset_name_list = ['celeba']
    dataset_name_list = ['vggface2']
    for dataset_name in dataset_name_list:
        print(os.getcwd())
        process = Preprocess(args, dataset_name)

        process.load_data()
        process.split_data()
        process.save_data()

        print(args['dataset_task'])
