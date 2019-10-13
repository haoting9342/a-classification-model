import os
import json
import time
import datetime as dt
import numpy as np
import pdb
import shutil
import logging
import sys
from data_processing import DataLoader
from QoE_model_rev4 import QoEModel_R4
from QoE_RF import QoE_RF

if __name__ == '__main__':

    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['Common']['save_dir']):
        os.makedirs(configs['Common']['save_dir'])

    # For every training, we will specify a folder to store all related files.
    folder_name = configs['Common']['ml'] + '_' + dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    main_path = os.path.join(configs['Common']['save_dir'], folder_name)
    os.makedirs(main_path)  # every thing will be stored in this folder, including model, log, config, and so on.

    # logging
    levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]
    level = logging.INFO
    if configs['Common']['logging_level'] == 'DEBUG':
        level = logging.DEBUG
    elif configs['Common']['logging_level'] == 'INFO':
        level = logging.INFO
    elif configs['Common']['logging_level'] == 'WARNING':
        level = logging.WARNING
    elif configs['Common']['logging_level'] == 'ERROR':
        level = logging.ERROR
    else:
        print('logging level input error !!!')

    log_file_name = os.path.join(main_path, 'log.txt')
    logging.basicConfig(level = level,
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(module)s - %(levelname)s :: %(message)s',
                        handlers=[
                            logging.FileHandler(log_file_name),
                            logging.StreamHandler(sys.stdout)
                        ])
    logger = logging.getLogger('QoE')
    logger.info(f"Log level {level}: {configs['Common']['logging_level']}")
    logger.info(f"All related files for this running are stored in {main_path}")

    # backup config
    cfg_old_path = 'config.json'
    cfg_new_path = os.path.join(main_path, cfg_old_path)
    shutil.copyfile(cfg_old_path, cfg_new_path)  # backup the configs.

    # start processing
    ml = configs['Common']['ml']
    data_path = configs['dp']['raw_data']
    label_mapping = configs['Common']['label_mapping']
    split_rate = configs['training']['val_split']
    group_size = configs['dp']['group_size']
    num_classes = configs['training']['n_classes']
    batch_size = configs['training']['batch_size']
    epochs = configs['training']['epochs']
    method = configs['training']['method']
    feature_set = configs['RF']['feature_set']
    feature_sel = configs['RF']['feature_selection']
    n_estimators = configs['RF']['n_estimators']
    max_depth = configs['RF']['max_depth']
    max_features = configs['RF']['max_features']
    criterion = configs['RF']['criterion']
    save_data = configs['dp']['save_data']
    normalization = configs['dp']['normalization']

    if save_data:
        processed_data_path = main_path
    else:
        processed_data_path = None

    if group_size == 0:
        group_size = None

    if ml=='training':
        logger.info("Start training...")
        data_processing = DataLoader()
        df = data_processing.run(data_path=data_path,
                                 label_mapping=label_mapping,
                                 group_size=group_size,
                                 save_data=processed_data_path,
                                 normalization=normalization)
        if method == 'RF':
            rf_model = QoE_RF(num_classes, label_mapping, main_path)
            rf_model.run(df, feature_set, feature_sel,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         max_features=max_features,
                         criterion=criterion)

        elif method == 'DNN':
            input_shape = df.shape[1]-1

            QoE_model = QoEModel_R4(input_shape, num_classes, label_mapping)
            QoE_model.run(df, split_rate,
                          batch_size=batch_size,
                          epochs=epochs,
                          save_path=main_path)
        else:
            logger.error(f"The method {method} is not supported yet!!!")
    elif ml=='testing':
        logger.info("Start testing...")
        data_processing = DataLoader()
        df = data_processing.run(data_path=data_path,
                                 label_mapping=label_mapping,
                                 group_size=group_size,
                                 save_data=processed_data_path,
                                 normalization=normalization)

        model_path = configs['testing']['model_path']
        model_type = os.path.splitext(model_path)[-1]

        if model_type=='.h5':
            if method!='DNN':
                logger.warning(f"Input method and the input model type are not match!! The input method is {method}, but the input model type is {model_type}. We will follow by {model_type} to perferm data processing!")
            dnn_model = QoEModel_R4(df.shape[1]-1, num_classes, label_mapping)
            dnn_model.inference(model_path=model_path, data=df, log_path=main_path)

            backup_model_path = os.path.join(main_path, 'dnn_model.h5')
            shutil.copyfile(model_path, backup_model_path)
        elif model_type=='.joblib':
            if method!='RF':
                logger.warning(
                    f"Input method and the input model type are not match!! The input method is {method}, but the input model type is {model_type}. We will follow by {model_type} to perferm data processing!")
            rf_model = QoE_RF(num_classes, label_mapping, main_path)
            rf_model.inference(path=model_path, data=df, features=feature_set)
            backup_model_path = os.path.join(main_path, 'rf_model.joblib')
            shutil.copyfile(model_path, backup_model_path)  # backup the configs.
        else:
            logger.error(f"The model type {model_type} is not support!")


