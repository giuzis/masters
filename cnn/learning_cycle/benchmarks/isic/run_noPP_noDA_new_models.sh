#!/bin/bash

# Declare an array of string with type
declare -a modelArray=("resnext101_32x8d" "efficientnet_b0" "efficientnet_b1" "efficientnet_b2" "efficientnet_b3" "efficientnet_b4" "tf_efficientnet_b5" "tf_efficientnet_b6" "vgg19" "seresnext101_32x8d" "resnest101e" "pnasnet5large" "densenet121")
declare -a optArray=("SGD" "Adam" "AdamW")
declare -a batch_size=(4 8 16 32 64)


python isic.py with _model_name=tf_efficientnet_b5 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2
python isic.py with _model_name=tf_efficientnet_b6 _lr_init=0.0001 _batch_size=4 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=densenet121 _lr_init=0.0001 _batch_size=32 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b0 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b1 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b2 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b3 _lr_init=0.0001 _batch_size=32 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b4 _lr_init=0.0001 _batch_size=16 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=resnest101e _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=seresnext101_32x8d _lr_init=0.0001 _batch_size=16 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=vgg19 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=SGD _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=resnext101_32x8d _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=SGD _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=pnasnet5large _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 

# sem o dull razor para ver como performa, pois tenho a impressao que ClaHE e dull_razor nao combinam

python isic.py with _model_name=tf_efficientnet_b5 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2
python isic.py with _model_name=tf_efficientnet_b6 _lr_init=0.0001 _batch_size=4 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=densenet121 _lr_init=0.0001 _batch_size=32 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b0 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b1 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b2 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b3 _lr_init=0.0001 _batch_size=32 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b4 _lr_init=0.0001 _batch_size=16 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=resnest101e _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=seresnext101_32x8d _lr_init=0.0001 _batch_size=16 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=vgg19 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=SGD _PP_enhancement=CLAHE _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=resnext101_32x8d _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=SGD _PP_enhancement=CLAHE _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=pnasnet5large _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE _PP_color_constancy=shades_of_gray _PP_crop_mode=cropped_images_folder _data_augmentation=2 

python isic.py with _model_name=tf_efficientnet_b5 _lr_init=0.0001 _batch_size=8 _optimizer=AdamW _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray 
python isic.py with _model_name=tf_efficientnet_b6 _lr_init=0.0001 _batch_size=4 _optimizer=AdamW _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray  
python isic.py with _model_name=densenet121 _lr_init=0.0001 _batch_size=32 _optimizer=Adam _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray  
python isic.py with _model_name=efficientnet_b0 _lr_init=0.0001 _batch_size=8 _optimizer=AdamW _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray  
python isic.py with _model_name=efficientnet_b1 _lr_init=0.0001 _batch_size=8 _optimizer=AdamW _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray  
python isic.py with _model_name=efficientnet_b2 _lr_init=0.0001 _batch_size=8 _optimizer=Adam _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray  
python isic.py with _model_name=efficientnet_b3 _lr_init=0.0001 _batch_size=32 _optimizer=Adam _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray  
python isic.py with _model_name=efficientnet_b4 _lr_init=0.0001 _batch_size=16 _optimizer=AdamW _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray  
python isic.py with _model_name=resnest101e _lr_init=0.0001 _batch_size=8 _optimizer=AdamW _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray  
python isic.py with _model_name=seresnext101_32x8d _lr_init=0.0001 _batch_size=16 _optimizer=Adam _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray  
python isic.py with _model_name=vgg19 _lr_init=0.0001 _batch_size=8 _optimizer=SGD _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray  
python isic.py with _model_name=resnext101_32x8d _lr_init=0.0001 _batch_size=8 _optimizer=SGD _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray  
python isic.py with _model_name=pnasnet5large _lr_init=0.0001 _batch_size=8 _optimizer=Adam _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _PP_color_constancy=shades_of_gray  
