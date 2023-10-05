#!/bin/bash

# Declare an array of string with type
declare -a modelArray=("resnext101_32x8d" "efficientnet_b0" "efficientnet_b1" "efficientnet_b2" "efficientnet_b3" "efficientnet_b4" "tf_efficientnet_b5" "tf_efficientnet_b6" "vgg19" "seresnext101_32x8d" "resnest101e" "pnasnet5large" "densenet121")
declare -a optArray=("SGD" "Adam" "AdamW")
declare -a batch_size=(8 16 64)


python isic.py with _model_name=tf_efficientnet_b5 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2


# AB
python isic.py with _model_name=tf_efficientnet_b5 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _data_augmentation=2
python isic.py with _model_name=tf_efficientnet_b6 _lr_init=0.0001 _batch_size=4 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _data_augmentation=2 
python isic.py with _model_name=tf_efficientnet_b6 _lr_init=0.0001 _batch_size=4 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _data_augmentation=2 
python isic.py with _model_name=densenet121 _lr_init=0.0001 _batch_size=32 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _data_augmentation=2 
python isic.py with _model_name=efficientnet_b0 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _data_augmentation=2 
python isic.py with _model_name=efficientnet_b1 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _data_augmentation=2 
python isic.py with _model_name=efficientnet_b2 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _data_augmentation=2 
python isic.py with _model_name=efficientnet_b3 _lr_init=0.0001 _batch_size=32 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _data_augmentation=2 
python isic.py with _model_name=efficientnet_b4 _lr_init=0.0001 _batch_size=16 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _data_augmentation=2 
python isic.py with _model_name=resnest101e _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _data_augmentation=2 
python isic.py with _model_name=seresnext101_32x8d _lr_init=0.0001 _batch_size=16 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _data_augmentation=2 
python isic.py with _model_name=vgg19 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=SGD _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _data_augmentation=2 
python isic.py with _model_name=resnext101_32x8d _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=SGD _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _data_augmentation=2 
python isic.py with _model_name=pnasnet5large _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _data_augmentation=2 

# AC
python isic.py with _model_name=tf_efficientnet_b5 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _data_augmentation=2
python isic.py with _model_name=tf_efficientnet_b6 _lr_init=0.0001 _batch_size=4 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=densenet121 _lr_init=0.0001 _batch_size=32 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=efficientnet_b0 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=efficientnet_b1 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=efficientnet_b2 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=efficientnet_b3 _lr_init=0.0001 _batch_size=32 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=efficientnet_b4 _lr_init=0.0001 _batch_size=16 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=resnest101e _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=seresnext101_32x8d _lr_init=0.0001 _batch_size=16 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=vgg19 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=SGD _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=resnext101_32x8d _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=SGD _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=pnasnet5large _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE _PP_hair_removal=dull_razor _data_augmentation=2 

# AD
python isic.py with _model_name=tf_efficientnet_b5 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_crop_mode=cropped_images_folder _data_augmentation=2
python isic.py with _model_name=tf_efficientnet_b6 _lr_init=0.0001 _batch_size=4 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=densenet121 _lr_init=0.0001 _batch_size=32 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b0 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b1 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b2 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b3 _lr_init=0.0001 _batch_size=32 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b4 _lr_init=0.0001 _batch_size=16 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=resnest101e _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=seresnext101_32x8d _lr_init=0.0001 _batch_size=16 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=vgg19 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=SGD _PP_enhancement=CLAHE _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=resnext101_32x8d _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=SGD _PP_enhancement=CLAHE _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=pnasnet5large _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE _PP_crop_mode=cropped_images_folder _data_augmentation=2 

# BC
python isic.py with _model_name=tf_efficientnet_b5 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2
python isic.py with _model_name=tf_efficientnet_b6 _lr_init=0.0001 _batch_size=4 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=densenet121 _lr_init=0.0001 _batch_size=32 _epochs=150 _early_stop=100 _optimizer=Adam _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=efficientnet_b0 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=efficientnet_b1 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=efficientnet_b2 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=Adam _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=efficientnet_b3 _lr_init=0.0001 _batch_size=32 _epochs=150 _early_stop=100 _optimizer=Adam _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=efficientnet_b4 _lr_init=0.0001 _batch_size=16 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=resnest101e _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=seresnext101_32x8d _lr_init=0.0001 _batch_size=16 _epochs=150 _early_stop=100 _optimizer=Adam _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=vgg19 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=SGD _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=resnext101_32x8d _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=SGD _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=pnasnet5large _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=Adam _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 

# CD
python isic.py with _model_name=tf_efficientnet_b5 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2
python isic.py with _model_name=tf_efficientnet_b6 _lr_init=0.0001 _batch_size=4 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=densenet121 _lr_init=0.0001 _batch_size=32 _epochs=150 _early_stop=100 _optimizer=Adam _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b0 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b1 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b2 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=Adam _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b3 _lr_init=0.0001 _batch_size=32 _epochs=150 _early_stop=100 _optimizer=Adam _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b4 _lr_init=0.0001 _batch_size=16 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=resnest101e _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=seresnext101_32x8d _lr_init=0.0001 _batch_size=16 _epochs=150 _early_stop=100 _optimizer=Adam _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=vgg19 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=SGD _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=resnext101_32x8d _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=SGD _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=pnasnet5large _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=Adam _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 

# ABC
python isic.py with _model_name=tf_efficientnet_b5 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2
python isic.py with _model_name=tf_efficientnet_b6 _lr_init=0.0001 _batch_size=4 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=densenet121 _lr_init=0.0001 _batch_size=32 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=efficientnet_b0 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=efficientnet_b1 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=efficientnet_b2 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=efficientnet_b3 _lr_init=0.0001 _batch_size=32 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=efficientnet_b4 _lr_init=0.0001 _batch_size=16 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=resnest101e _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=seresnext101_32x8d _lr_init=0.0001 _batch_size=16 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=vgg19 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=SGD _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=resnext101_32x8d _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=SGD _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 
python isic.py with _model_name=pnasnet5large _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE  _PP_color_constancy=shades_of_gray _PP_hair_removal=dull_razor _data_augmentation=2 

# ACD
python isic.py with _model_name=tf_efficientnet_b5 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE  _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2
python isic.py with _model_name=tf_efficientnet_b6 _lr_init=0.0001 _batch_size=4 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE  _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=densenet121 _lr_init=0.0001 _batch_size=32 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE  _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b0 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE  _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b1 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE  _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b2 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE  _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b3 _lr_init=0.0001 _batch_size=32 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE  _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=efficientnet_b4 _lr_init=0.0001 _batch_size=16 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE  _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=resnest101e _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=AdamW _PP_enhancement=CLAHE  _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=seresnext101_32x8d _lr_init=0.0001 _batch_size=16 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE  _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=vgg19 _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=SGD _PP_enhancement=CLAHE  _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=resnext101_32x8d _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=SGD _PP_enhancement=CLAHE  _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 
python isic.py with _model_name=pnasnet5large _lr_init=0.0001 _batch_size=8 _epochs=150 _early_stop=100 _optimizer=Adam _PP_enhancement=CLAHE  _PP_hair_removal=dull_razor _PP_crop_mode=cropped_images_folder _data_augmentation=2 