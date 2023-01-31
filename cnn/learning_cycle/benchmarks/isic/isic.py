
from raug.loader import get_data_loader
from raug.train import fit_model
from raug.eval import test_model
from my_model import set_model
import pandas as pd
import os
import torch.optim as optim
import torch.nn as nn
import torch
from aug_isic import ImgTrainTransform, ImgEvalTransform
import time
from sacred import Experiment
from sacred.observers import FileStorageObserver
from raug.utils.loader import get_labels_frequency


# Starting sacred experiment
ex = Experiment()

@ex.config
def cnfg():

    # Dataset variables
    _folder = 1
    _csv_path_train = "/home/a52550/Desktop/datasets/ISIC2017/train/ISIC-2017_Training_Part3_GroundTruth.csv"
    _imgs_folder_train = "/home/a52550/Desktop/datasets/ISIC2017/train/ISIC-2017_Training_Data/"

    _csv_path_validation = "/home/a52550/Desktop/datasets/ISIC2017/validation/ISIC-2017_Validation_Part3_GroundTruth.csv"
    _imgs_folder_validation = "/home/a52550/Desktop/datasets/ISIC2017/validation/ISIC-2017_Validation_Data_cropped/"
    _csv_path_test = "/home/a52550/Desktop/datasets/ISIC2017/test/ISIC-2017_Test_v2_Part3_GroundTruth.csv"
    _imgs_folder_test = "/home/a52550/Desktop/datasets/ISIC2017/test/ISIC-2017_Test_Data/"


    _batch_size = 4
    _epochs = 50

    # Training variables
    _best_metric = "accuracy"
    _pretrained = True
    _lr_init = 0.00001
    _sched_factor = 0.1
    _sched_min_lr = 1e-6
    _sched_patience = 10
    _early_stop = 15
    _weights = "frequency"
    _optimizer = 'SGD'

    _model_name = 'efficientnet_b0'
    _save_folder = "results/" + _model_name + "_fold_" + str(_folder) + "_" + str(
        time.time()).replace('.', '')

    # This is used to configure the sacred storage observer. In brief, it says to sacred to save its stuffs in
    # _save_folder. You don't need to worry about that.
    SACRED_OBSERVER = FileStorageObserver(_save_folder)
    ex.observers.append(SACRED_OBSERVER)

@ex.automain
def main (_csv_path_train, _imgs_folder_train, _csv_path_validation, _imgs_folder_validation, _csv_path_test, _imgs_folder_test, _lr_init, _sched_factor,
          _sched_min_lr, _sched_patience, _batch_size, _epochs, _early_stop, _weights, _model_name,
          _save_folder, _best_metric, _optimizer):

    _metric_options = {
        'save_all_path': os.path.join(_save_folder, "best_metrics"),
        'pred_name_scores': 'predictions_best_test.csv',
        'normalize_conf_matrix': True}
    _checkpoint_best = os.path.join(_save_folder, 'best-checkpoint/best-checkpoint.pth')

    print("-" * 50)
    print("- Loading validation data...")
    val_csv_folder = pd.read_csv(_csv_path_validation)
    train_csv_folder = pd.read_csv(_csv_path_train)
    
    ser_lab_freq = get_labels_frequency(train_csv_folder, "category", "image_id")
    _labels_name = ser_lab_freq.index.values
    _freq = ser_lab_freq.values
    ####################################################################################################################

    model = set_model(_model_name, len(_labels_name))

    # Loading validation data
    val_imgs_id = val_csv_folder['image_id'].values
    print("-- Using {} images".format(val_imgs_id.size))
    val_imgs_path = ["{}{}.jpg".format(_imgs_folder_validation, img_id) for img_id in val_imgs_id]
    val_labels = val_csv_folder['category'].values
    val_meta_data = None

    print("-- val_data_loader starting...")
    val_data_loader = get_data_loader (val_imgs_path, val_labels, val_meta_data, transform=ImgEvalTransform(size=model.default_cfg['input_size'][1:], normalization=(model.default_cfg['mean'], model.default_cfg['std'])),
                                       batch_size=_batch_size, shuf=True, num_workers=16, pin_memory=True)
    print("-- Validation partition loaded with {} images".format(len(val_data_loader)*_batch_size))

    print("- Loading training data...")
    train_imgs_id = train_csv_folder['image_id'].values
    print("-- Using {} images".format(train_imgs_id.size))
    train_imgs_path = ["{}{}.jpg".format(_imgs_folder_train, img_id) for img_id in train_imgs_id]
    train_labels = train_csv_folder['category'].values
    train_meta_data = None

    train_data_loader = get_data_loader (train_imgs_path, train_labels, train_meta_data, transform=ImgEvalTransform(size=model.default_cfg['input_size'][1:], normalization=(model.default_cfg['mean'], model.default_cfg['std'])),
                                       batch_size=_batch_size, shuf=True, num_workers=16, pin_memory=True)
    print("-- Training partition loaded with {} images".format(len(train_data_loader)*_batch_size))

    print("-"*50)
    ####################################################################################################################

    print("- Loading", _model_name)

    ####################################################################################################################
    if _weights == 'frequency':
        _weights = (_freq.sum() / _freq).round(3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(_weights).cuda())
    if _optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=_lr_init)
    else:
        optimizer = optim.SGD(model.parameters(), lr=_lr_init)
    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=_sched_factor, min_lr=_sched_min_lr,
                                                                    patience=_sched_patience)
    ####################################################################################################################

    print("- Starting the training phase...")
    print("-" * 50)
    fit_model (model, train_data_loader, val_data_loader, class_names=_labels_name, optimizer=optimizer, loss_fn=loss_fn, epochs=_epochs,
               epochs_early_stop=_early_stop, save_folder=_save_folder, initial_model=None,
               device=None, schedule_lr=scheduler_lr, config_bot=None, model_name="CNN", resume_train=False,
               history_plot=True, val_metrics=["accuracy", "topk_accuracy", "balanced_accuracy",  "conf_matrix", "plot_conf_matrix",
                                  "precision_recall_report", "auc_and_roc_curve", "auc"]
                                  , best_metric=_best_metric)
    # ####################################################################################################################

    # Testing the validation partition
    print("- Evaluating the validation partition...")
    test_model (model, val_data_loader, checkpoint_path=_checkpoint_best, loss_fn=loss_fn, save_pred=True,
                partition_name='eval', metrics_to_comp='all', class_names=_labels_name, metrics_options=_metric_options,
                apply_softmax=True, verbose=False)
    ####################################################################################################################

    ####################################################################################################################

    if _csv_path_test is not None:
        print("- Loading test data...")
        csv_test = pd.read_csv(_csv_path_test)
        test_imgs_id = csv_test['image_id'].values
        test_imgs_path = ["{}/{}.jpg".format(_imgs_folder_test, img_id) for img_id in test_imgs_id]
        test_labels = csv_test['category'].values

        _metric_options = {
            'save_all_path': os.path.join(_save_folder, "test_pred"),
            'pred_name_scores': 'predictions.csv',
        }
        test_data_loader = get_data_loader(test_imgs_path, test_labels, transform=ImgEvalTransform(size=model.default_cfg['input_size'][1:], normalization=(model.default_cfg['mean'], model.default_cfg['std'])),
                                           batch_size=_batch_size, shuf=False, num_workers=16, pin_memory=True)
        print("-" * 50)

        # Testing the test partition
        print("\n- Evaluating the testing partition...")
        test_model(model, test_data_loader, metrics_to_comp='all',
                   class_names=_labels_name, metrics_options=_metric_options, save_pred=False, verbose=True)
    ####################################################################################################################