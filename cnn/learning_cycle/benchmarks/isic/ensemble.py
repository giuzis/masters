import os
import numpy as np
from raug.metrics import Metrics, accuracy
from raug.utils.loader import get_labels_frequency
import pandas as pd

# Especificar o caminho para a pasta "best_results"
folder_path = "best_results"
_csv_path_test = "/home/a52550/Desktop/datasets/ISIC2017/test/ISIC-2017_Test_v2_Part3_GroundTruth.csv"
true_labels = pd.read_csv(_csv_path_test).sort_values(by=['image_id'])['category']

# Inicializar as variáveis para armazenar as previsões dos modelos
ensemble_preds_best = None
ensemble_preds_last = None
num_models = 0
ser_lab_freq = get_labels_frequency(_csv_path_test, "category", "image_id")
_labels_name = ser_lab_freq.index.values #np.reshape(ser_lab_freq.index.values, (ser_lab_freq.index.values.shape[0], 1))

df_best = pd.DataFrame()
df_last = pd.DataFrame()

# Percorrer as pastas de cada experimento
for experiment_folder in os.listdir(folder_path):

    if ".csv" in experiment_folder:
        continue

    print("Experimento: ", experiment_folder)

    _metric_options_best = {
        'save_all_path': os.path.join(folder_path, experiment_folder, 'ensemble_3_best_average/best'),
        'pred_name_scores': 'predictions.csv',
    }
    _metric_options_last = {
        'save_all_path': os.path.join(folder_path, experiment_folder, 'ensemble_3_best_average/last'),
        'pred_name_scores': 'predictions.csv',
    }


    experiment_path = os.path.join(folder_path, experiment_folder)
    metrics_best = Metrics (["accuracy", "topk_accuracy", "balanced_accuracy", "conf_matrix" "plot_conf_matrix", "precision_recall_report", "auc_and_roc_curve", "auc"] , _labels_name, _metric_options_best)
    metrics_last = Metrics (["accuracy", "topk_accuracy", "balanced_accuracy", "conf_matrix" "plot_conf_matrix", "precision_recall_report", "auc_and_roc_curve", "auc"] , _labels_name, _metric_options_last)

    ensemble_preds_best = None
    ensemble_preds_last = None

    experiment_folders = os.listdir(experiment_path)
    
    # Cria path para o experimento atual
    if 'ensemble_average' in experiment_folders:
        experiment_folders.remove('ensemble_average')
    if 'ensemble_3_best_average' in experiment_folders:
        experiment_folders.remove('ensemble_3_best_average')
    else:
        os.mkdir(os.path.join(folder_path, experiment_folder, 'ensemble_3_best_average'))
        os.mkdir(os.path.join(_metric_options_best['save_all_path']))
        os.mkdir(os.path.join(_metric_options_last['save_all_path']))


    try:
        best_balanced_accuracy = 0
        best_metric_ensemble = None

        # Percorrer as pastas de cada modelo
        for i1 in range(len(experiment_folders)-1):
            for i2 in range(i1+1, len(experiment_folders)-1):
                for i3 in range(i2+1, len(experiment_folders)-1):
                    model_folders = [experiment_folders[i1], experiment_folders[i2], experiment_folders[i3]]

                    for model_folder in model_folders:

                        model_path = os.path.join(experiment_path, model_folder)

                        # Carregar as previsões do modelo atual
                        model_preds_path_best = os.path.join(model_path, "test_pred_best/predictions.csv")
                        model_preds_path_last = os.path.join(model_path, "test_pred_last/predictions.csv")

                        model_preds_best = pd.read_csv(model_preds_path_best).sort_values(by=['image'])
                        model_preds_last = pd.read_csv(model_preds_path_last).sort_values(by=['image'])

                        images_id = model_preds_best[['image']]
                        model_preds_best = model_preds_best[['0','1','2']]
                        model_preds_last = model_preds_last[['0','1','2']]

                        # Adicionar as previsões do modelo atual ao ensemble
                        if ensemble_preds_best is None:
                            ensemble_preds_best = model_preds_best
                        else:
                            ensemble_preds_best += model_preds_best

                        if ensemble_preds_last is None:
                            ensemble_preds_last = model_preds_last
                        else:
                            ensemble_preds_last += model_preds_last

                        num_models += 1

                    # Calcular a média das previsões dos modelos
                    ensemble_preds_best /= num_models
                    ensemble_preds_last /= num_models

                    metrics_best.update_scores(true_labels, ensemble_preds_best.to_numpy(), images_id.to_numpy())
                    metrics_last.update_scores(true_labels, ensemble_preds_last.to_numpy(), images_id.to_numpy())

                    metrics_best.compute_metrics()
                    metrics_last.compute_metrics()

                    # metrics_best.save_scores()
                    # metrics_last.save_scores()

                    if metrics_best.metrics_values['balanced_accuracy'] > best_balanced_accuracy:
                        best_balanced_accuracy = metrics_best.metrics_values['balanced_accuracy']
                        best_metric_ensemble = metrics_best
                        best_experiment = experiment_folder

                        ensemble_data = {
                            'folder_name': experiment_folder,
                            'ensemble_3_best': model_folders[0].split('fold')[0] + "_" + model_folders[1].split('fold')[0] + "_" + model_folders[2].split('fold')[0]
                        }
                        dict_best = {**ensemble_data, **metrics_best.metrics_values}
                        dict_last = {**ensemble_data, **metrics_last.metrics_values}

                        del dict_best['auc_and_roc_curve']
                        del dict_best['precision_recall_report']

                        del dict_last['auc_and_roc_curve']
                        del dict_last['precision_recall_report']

                    # df_best = df_best.append(pd.DataFrame(dict_best, columns=dict_best.keys(), index=[0]), ignore_index=True)
                    # df_last = df_last.append(pd.DataFrame(dict_last, columns=dict_last.keys(), index=[0]), ignore_index=True)

        best_metrics_ensemble.save_scores()

        df_best = df_best.append(pd.DataFrame(dict_best, columns=dict_best.keys(), index=[0]), ignore_index=True)
        df_last = df_last.append(pd.DataFrame(dict_last, columns=dict_last.keys(), index=[0]), ignore_index=True)

    except Exception as e:
        print("Erro no experimento ", experiment_folder)
        print(e.with_traceback())



df_best.to_csv(os.path.join(folder_path, 'ensemble_3_best_average_best.csv'), index=False)
df_last.to_csv(os.path.join(folder_path, 'ensemble_3_best_average_last.csv'), index=False)
