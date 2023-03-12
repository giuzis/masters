
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
from timm.optim import optim_factory
from types import SimpleNamespace


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

    _csv_path_all_metrics = "results/all_metrics.csv"


    # Training variables
    _batch_size = 32
    _epochs = 50
    _best_metric = "accuracy"
    _pretrained = True
    _lr_init = 0.001
    _sched_factor = 0.1
    _sched_min_lr = 1e-6
    _sched_patience = 10
    _early_stop = 15
    _weights = "frequency"
    _optimizer = 'Adam' # 'SGD', 'Adam', 'AdamW', 'Nadam', 'Radam', 'AdamP', 'Lookahead_Adam', 'Lookahead_AdamW', 'Lookahead_Nadam', 'Lookahead_Radam', 'Lookahead_AdamP'
    _data_augmentation = False
    _PP_enhancement = None
    _PP_hair_removal = None
    _PP_color_constancy = None
    _PP_denoising = None
    _PP_normalization = True
    _PP_crop_mode = None
    _PP_resizing = True

    _model_name = 'senet154'
    # _save_folder = "results/" + _model_name + "_fold_" + str(_folder) + "_" + str(time.time()).replace('.', '')
    _save_folder = f"results/{_model_name}_" +\
        f"fold-{_folder}_" +\
        f"lrinit-{_lr_init}_" +\
        f"batchsize-{_batch_size}_" +\
        f"optimizer-{_optimizer}_" +\
        f"maxepochs-{_epochs}_" +\
        f"DA-{_data_augmentation}_" +\
        f"PPen-{_PP_enhancement}_" +\
        f"PPha-{_PP_hair_removal}_" +\
        f"PPco-{_PP_color_constancy}_" +\
        f"PPde-{_PP_denoising}_" +\
        f"PPno-{_PP_normalization}_" +\
        f"PPcr-{_PP_crop_mode}_" +\
        f"PPre-{_PP_resizing}_" +\
        f"{str(time.time()).replace('.', '')}"

    # This is used to configure the sacred storage observer. In brief, it says to sacred to save its stuffs in
    # _save_folder. You don't need to worry about that.
    SACRED_OBSERVER = FileStorageObserver(_save_folder)
    ex.observers.append(SACRED_OBSERVER)

@ex.automain
def main (_csv_path_train, _imgs_folder_train, _csv_path_validation, _imgs_folder_validation, 
          _csv_path_test, _imgs_folder_test, _lr_init, _sched_factor, _sched_min_lr, _sched_patience, 
          _batch_size, _epochs, _early_stop, _weights, _model_name, _save_folder, _best_metric, _optimizer, 
          _csv_path_all_metrics, _data_augmentation, _PP_enhancement, _PP_hair_removal, _PP_color_constancy,
          _PP_denoising, _PP_normalization, _PP_crop_mode, _PP_resizing):

    _metric_options = {
        'save_all_path': os.path.join(_save_folder, "best_metrics"),
        'pred_name_scores': 'predictions_best_test.csv',
        'normalize_conf_matrix': True}
    _checkpoint_best = os.path.join(_save_folder, 'best-checkpoint/best-checkpoint.pth')
    _checkpoint_last = os.path.join(_save_folder, 'last-checkpoint/last-checkpoint.pth')

    _all_metrics = ["accuracy", "topk_accuracy", "balanced_accuracy",  "conf_matrix", 
                    "plot_conf_matrix", "precision_recall_report", "auc_and_roc_curve", "auc"]

    print("-" * 50)
    print("- Loading validation data...")
    val_csv_folder = pd.read_csv(_csv_path_validation)
    train_csv_folder = pd.read_csv(_csv_path_train)
    try:
        all_metrics_df = pd.read_csv(_csv_path_all_metrics)
    except:
        all_metrics_df = pd.DataFrame()

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

    val_data_loader = get_data_loader (val_imgs_path, val_labels, val_meta_data, 
                                       transform=ImgEvalTransform(size=model.default_cfg['input_size'][1:], 
                                       normalization=(model.default_cfg['mean'], model.default_cfg['std'])),
                                       batch_size=_batch_size, shuf=True, pin_memory=True)
    print("-- Validation partition loaded with {} images".format(len(val_data_loader)*_batch_size))

    print("- Loading training data...")
    train_imgs_id = train_csv_folder['image_id'].values
    print("-- Using {} images".format(train_imgs_id.size))
    train_imgs_path = ["{}{}.jpg".format(_imgs_folder_train, img_id) for img_id in train_imgs_id]
    train_labels = train_csv_folder['category'].values
    train_meta_data = None
    if _data_augmentation == True or _data_augmentation == "True":
        train_transform = ImgTrainTransform(size=model.default_cfg['input_size'][1:], 
                                         normalization=(model.default_cfg['mean'], model.default_cfg['std']),
                                         type=1, crop_mode = _PP_crop_mode)

    train_data_loader = get_data_loader (train_imgs_path, train_labels, train_meta_data, 
                                         transform=ImgEvalTransform(size=model.default_cfg['input_size'][1:], 
                                         normalization=(model.default_cfg['mean'], model.default_cfg['std'])),
                                         batch_size=_batch_size, shuf=True, pin_memory=True)
    print("-- Training partition loaded with {} images".format(len(train_data_loader)*_batch_size))

    print("-"*50)
    ####################################################################################################################

    print("- Loading", _model_name)
    print("- Using optimizer", _optimizer)
    print("- Input size", model.default_cfg['input_size'][1:])

    ####################################################################################################################
    if _weights == 'frequency':
        _weights = (_freq.sum() / _freq).round(3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(_weights).cuda())
    # if _optimizer == 'adam':
    #     optimizer = optim.Adam(model.parameters(), lr=_lr_init)
    # else:
    #     optimizer = optim.SGD(model.parameters(), lr=_lr_init)
    args = SimpleNamespace()
    args.opt = _optimizer
    args.lr = _lr_init
    args.weight_decay = 0 if _optimizer != 'AdamW' else 1e-2
    args.momentum = 0.9


    optimizer = optim_factory.create_optimizer(args, model)
    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=_sched_factor, min_lr=_sched_min_lr,
                                                                    patience=_sched_patience)
    ####################################################################################################################

    print("-" * 50)
    print("- Starting the training phase...")
    print("-" * 50)
    _last_epoch, _best_metric_value, _best_epoch = fit_model (model, train_data_loader, val_data_loader, class_names=_labels_name, 
                                                              optimizer=optimizer, loss_fn=loss_fn, epochs=_epochs, epochs_early_stop=_early_stop, 
                                                              save_folder=_save_folder, initial_model=None, device=None, schedule_lr=scheduler_lr,
                                                              config_bot=None, model_name="CNN", resume_train=False, history_plot=True, 
                                                              val_metrics=_all_metrics, best_metric=_best_metric)
    # ####################################################################################################################

    # Testing the validation partition
    print("-" * 50)
    print("- Evaluating the validation partition...")
    print("-" * 50)

    all_metrics_dict = {
        "folder":_save_folder,
        "model_name":_model_name,
        "batch_size":_batch_size,
        "epochs":_epochs,
        "lr_init":_lr_init,
        "sched_factor":_sched_factor,
        "sched_min_lr":_sched_min_lr,
        "sched_patience":_sched_patience,
        "early_stop":_early_stop,
        "optimizer":_optimizer,
        "partition":"validation",
        "best_metric_train":_best_metric,
        "best_metric_value_train": _best_metric_value,
        "best_epoch_train": _best_epoch,
        "last_epoch_train": _last_epoch,
        "data_augmentation": _data_augmentation,
        "PP_enhancement": _PP_enhancement,
        "PP_hair_removal": _PP_hair_removal,
        "PP_color_constancy": _PP_color_constancy,
        "PP_denoising": _PP_denoising,
        "PP_normalization": _PP_normalization,
        "PP_crop_mode": _PP_crop_mode,
        "PP_resizing": _PP_resizing,
    }

    validation_metrics = test_model (model, val_data_loader, checkpoint_path=_checkpoint_best, loss_fn=loss_fn, save_pred=True,
                partition_name='eval', metrics_to_comp=_all_metrics, class_names=_labels_name, metrics_options=_metric_options,
                apply_softmax=True, verbose=False)
    
    del validation_metrics['conf_matrix']
    del validation_metrics['auc_and_roc_curve']
    del validation_metrics['precision_recall_report']

    new_dict = {**all_metrics_dict, **validation_metrics}
    
    all_metrics_df = all_metrics_df.append(pd.DataFrame(new_dict, columns=new_dict.keys(), index=[0]), ignore_index=True)
    ####################################################################################################################

    ####################################################################################################################

    if _csv_path_test is not None:
        print("- Loading test data...")
        csv_test = pd.read_csv(_csv_path_test)
        test_imgs_id = csv_test['image_id'].values
        test_imgs_path = ["{}/{}.jpg".format(_imgs_folder_test, img_id) for img_id in test_imgs_id]
        test_labels = csv_test['category'].values

        all_metrics_dict = {
            "folder":_save_folder,
            "model_name":_model_name,
            "batch_size":_batch_size,
            "epochs":_epochs,
            "lr_init":_lr_init,
            "sched_factor":_sched_factor,
            "sched_min_lr":_sched_min_lr,
            "sched_patience":_sched_patience,
            "early_stop":_early_stop,
            "optimizer":_optimizer,
            "partition":"test",
            "best_metric_train":_best_metric,
            "best_metric_value_train": _best_metric_value,
            "best_epoch_train": _best_epoch,
            "last_epoch_train": _last_epoch,
            "data_augmentation": _data_augmentation,
            "PP_enhancement": _PP_enhancement,
            "PP_hair_removal": _PP_hair_removal,
            "PP_color_constancy": _PP_color_constancy,
            "PP_denoising": _PP_denoising,
            "PP_normalization": _PP_normalization,
            "PP_crop_mode": _PP_crop_mode,
            "PP_resizing": _PP_resizing,
        }

        _metric_options = {
            'save_all_path': os.path.join(_save_folder, "test_pred_best"),
            'pred_name_scores': 'predictions.csv',
        }
        test_data_loader = get_data_loader(test_imgs_path, test_labels, 
                                            transform=ImgEvalTransform(size=model.default_cfg['input_size'][1:], 
                                            normalization=(model.default_cfg['mean'], model.default_cfg['std'])),
                                            batch_size=_batch_size, shuf=False, pin_memory=True)
        print("-" * 50)

        # Testing the test partition
        print("-" * 50)
        print("\n- Evaluating the testing partition for the best accuracy epoch model...")
        print("-" * 50)
        test_metrics = test_model (model, test_data_loader, checkpoint_path=_checkpoint_best, save_pred=True,
                metrics_to_comp=_all_metrics, class_names=_labels_name, metrics_options=_metric_options,
                verbose=True)
        
        del test_metrics['conf_matrix']
        del test_metrics['auc_and_roc_curve']
        del test_metrics['precision_recall_report']

        new_dict = {**all_metrics_dict, **test_metrics}
        
        all_metrics_df = all_metrics_df.append(pd.DataFrame(new_dict, columns=new_dict.keys(), index=[0]), ignore_index=True)

        all_metrics_df.to_csv(_csv_path_all_metrics, index=False)
        
        _metric_options = {
            'save_all_path': os.path.join(_save_folder, "test_pred_last"),
            'pred_name_scores': 'predictions.csv',
        }
        
        test_metrics = test_model (model, test_data_loader, checkpoint_path=_checkpoint_last, save_pred=True,
                metrics_to_comp=_all_metrics, class_names=_labels_name, metrics_options=_metric_options,
                verbose=True)
        
        all_metrics_dict = {
            "folder":_save_folder,
            "model_name":_model_name,
            "batch_size":_batch_size,
            "epochs":_epochs,
            "lr_init":_lr_init,
            "sched_factor":_sched_factor,
            "sched_min_lr":_sched_min_lr,
            "sched_patience":_sched_patience,
            "early_stop":_early_stop,
            "optimizer":_optimizer,
            "partition":"test-last",
            "best_metric_train":_best_metric,
            "best_metric_value_train": _best_metric_value,
            "best_epoch_train": _best_epoch,
            "last_epoch_train": _last_epoch,
            "data_augmentation": _data_augmentation,
            "PP_enhancement": _PP_enhancement,
            "PP_hair_removal": _PP_hair_removal,
            "PP_color_constancy": _PP_color_constancy,
            "PP_denoising": _PP_denoising,
            "PP_normalization": _PP_normalization,
            "PP_crop_mode": _PP_crop_mode,
            "PP_resizing": _PP_resizing
        }
        
        del test_metrics['conf_matrix']
        del test_metrics['auc_and_roc_curve']
        del test_metrics['precision_recall_report']

        new_dict = {**all_metrics_dict, **test_metrics}
        
        all_metrics_df = all_metrics_df.append(pd.DataFrame(new_dict, columns=new_dict.keys(), index=[0]), ignore_index=True)

        all_metrics_df.to_csv(_csv_path_all_metrics, index=False)
    ####################################################################################################################