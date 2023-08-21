# dds_drone_ws

1. Combine and copy all image to C:/GitHub/dds_drone_ws/workspace/training_dds/images/
- ensure the image file is match with xml file
- ensure there is no corrupted image - run script data_cleaning.py

cd c:\github\dds_drone_ws\script\preprocessing
python data_cleaning.py


2. partition dataset for training and test 

python partition_dataset.py -x -i [PATH_TO_IMAGES_FOLDER] -r 0.1

# For example
run at ./scripts/preprocessing/
python partition_dataset.py -x -i /content/images -r 0.2 -o /content/


3. generate the tfrecord file

# Create train data:
python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/train -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/train.record

# Create test data:
python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/test -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/test.record

# For example
python generate_tfrecord.py -x C:/GitHub/dds_drone_ws/workspace/training_dds/images/train -l C:/GitHub/dds_drone_ws/workspace/training_dds/annotations/label_map.pbtxt -o C:/GitHub/dds_drone_ws/workspace/training_dds/annotations/train.record
python generate_tfrecord.py -x C:/GitHub/dds_drone_ws/workspace/training_dds/images/test  -l C:/GitHub/dds_drone_ws/workspace/training_dds/annotations/label_map.pbtxt -o C:/GitHub/dds_drone_ws/workspace/training_dds/annotations/test.record

4. Run the training 
- update pipeline config with number of category
- delete previous model at C:\GitHub\dds_drone_ws\workspace\training_dds\models\my_ssd_resnet50_v1_fpn\
- only left the pipeline config

python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config

5. To view the progress
Open new cmd enviroment tf38, at directory trainingdds, run

tensorboard --logdir=models/my_ssd_resnet50_v1_fpn

* tips
Following what people have said online, it seems that it is advisable to allow you model to reach a 
TotalLoss of at least 2 (ideally 1 and lower) if you want to achieve “fair” detection results. 
Obviously, lower TotalLoss is better, however very low TotalLoss should be avoided, as the model may end up overfitting the dataset, 
meaning that it will perform poorly when applied to images outside the dataset. 


6. export the inference
python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_ssd_resnet50_v1_fpn\pipeline.config --trained_checkpoint_dir .\models\my_ssd_resnet50_v1_fpn\ --output_directory .\exported-models

7. evaluate using video
python evaluate_video.py
