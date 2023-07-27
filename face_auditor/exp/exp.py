import datetime
import logging
from re import S
import os
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
        self.shadow_dataset_name = args['shadow_dataset_name']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.load_data()

    def load_data(self):
        self.logger.info('loading data')

        # Same dataset
        self.data_store = DataStore(self.args)
        if self.dataset_name == self.shadow_dataset_name:
            self.target_train_dset, self.shadow_train_dset, \
            self.target_train_mem_dset, self.target_train_nonmem_dset, \
            self.shadow_train_mem_dset, self.shadow_train_nonmem_dset, \
            self.target_test_dset, self.shadow_test_dset = \
            self.data_store.load_data()
        else:
            # Target model-related data
            self.target_train_dset, _, \
            self.target_train_mem_dset, self.target_train_nonmem_dset, \
            _, _, \
            self.target_test_dset, _ = \
            self.data_store.load_data()
            # Shadow model-related data
            _, self.shadow_train_dset, \
            _, _, \
            self.shadow_train_mem_dset, self.shadow_train_nonmem_dset, \
            _, self.shadow_test_dset = \
            self.data_store.load_data(dataset=self.shadow_dataset_name, ratio=self.args['victim_ratio_shadow'])

    def write_results(self, upload_data):
        self.logger.info("saving results")

        upload_data_extra = {
            'dataset': self.dataset_name,
            'shadow_dataset_name': self.shadow_dataset_name,
            'dataset_task': self.args['dataset_task'],
            'image_size': self.args['image_size'],
            'target_model': self.args['target_model'],
            'feature_extractor': self.args['feature_extractor'],
            'train_num_task': self.args['train_num_task'],
            'is_dp_defense': self.args['is_dp_defense'],
            'is_noise_defense': self.args['is_noise_defense'],
            'is_adv_defense': self.args['is_adv_defense'],
            'is_memguard_defense': self.args['is_memguard_defense'],
            'victim_ratio': self.args['victim_ratio'],
            'victim_ratio_shadow': self.args['victim_ratio_shadow'],
            'test_ratio': self.args['test_ratio']
        }

        for k, v in upload_data_extra.items():
            if k not in upload_data:
                upload_data[k] = v
        
        # Make sure RESULT_PATH folder exists
        if not os.path.exists(config.RESULT_PATH):
            os.makedirs(config.RESULT_PATH)

        with open(config.RESULT_PATH + self.args['database_table_name']+'.json', 'a') as f:
            json.dump(upload_data, f, sort_keys=True, default=str)
