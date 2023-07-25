import datetime
import logging
from re import S
import time

import torch
import json

import face_auditor.config as config
from face_auditor.lib_dataset.data_store import DataStore


class Exp:
    def __init__(self, args):
        self.logger = logging.getLogger(__name__)

        self.args = args
        self.start_time = datetime.datetime.now()
        self.dataset_name = args['dataset_name']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.load_data()

    def load_data(self):
        self.logger.info('loading data')

        self.data_store = DataStore(self.args)

        self.target_train_dset, self.shadow_train_dset, \
        self.target_train_mem_dset, self.target_train_nonmem_dset, \
        self.shadow_train_mem_dset, self.shadow_train_nonmem_dset, \
        self.target_test_dset, self.shadow_test_dset = \
        self.data_store.load_data()

    def write_results(self, upload_data):
        self.logger.info("saving results")

        upload_data_extra = {
            'dataset': self.dataset_name,
            'shadow_dataset_name': self.args['shadow_dataset_name'],
            'dataset_task': self.args['dataset_task'],
            'image_size': self.args['image_size'],
            'target_model': self.args['target_model'],
            'feature_extractor': self.args['feature_extractor'],
            'train_num_task': self.args['train_num_task'],
            'is_dp_defense': self.args['is_dp_defense'],
            'is_noise_defense': self.args['is_noise_defense'],
            'is_adv_defense': self.args['is_adv_defense'],
            'is_memguard_defense': self.args['is_memguard_defense']
        }

        for k, v in upload_data_extra.items():
            if k not in upload_data:
                upload_data[k] = v

        with open(config.RESULT_PATH + self.args['database_table_name']+'.json', 'a') as f:
            json.dump(upload_data, f, sort_keys=True, default=str)
