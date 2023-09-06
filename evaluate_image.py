import os

working_dir =  os.getcwd() + "\\exported-models\\"
model_version = os.listdir(working_dir)[len(os.listdir(working_dir)) -1]
working_dir = working_dir + model_version 
working__model_dir = working_dir + "\\model\\"

PATH_TO_CKPT = working__model_dir + "checkpoint\\"
PATH_TO_CFG = working__model_dir+ 'pipeline.config'
PATH_TO_LABELS = working_dir + "\\label_map.pbtxt"

print(PATH_TO_CKPT)
print(PATH_TO_CFG)
print(PATH_TO_LABELS)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
import cv2
import numpy as np
import datetime
import re
import matplotlib.pyplot as plt


def evaluate_folder(image_folder, min_score_thresh):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    total_drone = 0
    total_notdrone = 0
    total_image = 0

    # Loop through the image files and display them
    for image_file in image_files:
        total_image = total_image + 1

        image_path = os.path.join(image_folder, image_file)

        # Read the image using OpenCV
        image = cv2.imread(image_path)
        image_np = image.copy()
        image_np_expanded = np.expand_dims(image_np, axis=0)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        # min_score_thresh = .25

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=min_score_thresh,
            agnostic_mode=False)


        boxes = detections['detection_boxes'][0].numpy()
        classes = (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)
        scores=detections['detection_scores'][0].numpy()


        is_detect = False
        largestBoundingBox = None
        img_brg = image_np_with_detections
        class_name = ''
        percentage = 0

        if boxes.shape[0] == 0:
            print(f'{image_file} - NO DETECTION')
        else:
            for i in range(boxes.shape[0]):
                display_str = ''
                if scores is None or scores[i] > min_score_thresh:
                    box = tuple(boxes[i].tolist())  
                    ymin, xmin, ymax, xmax = box
                    im_width, im_height = image_np_with_detections.shape[1], image_np_with_detections.shape[0] 
                    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                                ymin * im_height, ymax * im_height)

                    class_temp = category_index[classes[i]]['name']
                    display_str = str(class_temp)
                    percentage_temp = round(100*scores[i])
                    display_str = '{}: {}%'.format(display_str, percentage_temp)
                    # print(f'{image_file} - [{class_temp}] [{display_str}]')
                    if class_temp == 'drone':
                        total_drone = total_drone + 1
                    if class_temp == 'notdrone':
                        total_notdrone = total_notdrone + 1

    


    # splash = cv2.resize(image_np_with_detections, (800, 600))
    # cv2.imshow(image_file, splash)
    # cv2.waitKey(0)  # Wait for a key press
    # cv2.destroyAllWindows()  # Close the window after key press

    # # Convert BGR to RGB (OpenCV uses BGR by default)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # # Display the image using Matplotlib
    # plt.imshow(image_rgb)
    # plt.title(image_file)
    # plt.axis('off')  # Turn off axes
    # plt.show()

    print(f'Model [{model_version}] Threshold [{min_score_thresh}]  Image [{total_image}] Drone [{total_drone}] NotDrone [{total_notdrone}]')


# Path to the folder containing images
# evaluate_folder('evaluation/drone',.25)
# evaluate_folder('evaluation/notdrone',.25)

# evaluate_folder('evaluation/drone',.3)
# evaluate_folder('evaluation/notdrone',.3)

# evaluate_folder('evaluation/drone',.35)
# evaluate_folder('evaluation/notdrone',.35)

# evaluate_folder('evaluation/drone',.4)
# evaluate_folder('evaluation/notdrone',.4)

# evaluate_folder('evaluation/drone',.45)
# evaluate_folder('evaluation/notdrone',.45)

# evaluate_folder('evaluation/drone',.5)
# evaluate_folder('evaluation/notdrone',.5)

# evaluate_folder('evaluation/drone',.55)
# evaluate_folder('evaluation/notdrone',.55)


# evaluate_folder('evaluation/drone',.6)
# evaluate_folder('evaluation/notdrone',.6)


evaluate_folder('evaluation/drone',.65)
evaluate_folder('evaluation/notdrone',.65)

evaluate_folder('evaluation/drone',.7)
evaluate_folder('evaluation/notdrone',.7)

evaluate_folder('evaluation/drone',.75)
evaluate_folder('evaluation/notdrone',.75)

evaluate_folder('evaluation/drone',.8)
evaluate_folder('evaluation/notdrone',.8)

evaluate_folder('evaluation/drone',.85)
evaluate_folder('evaluation/notdrone',.85)

evaluate_folder('evaluation/drone',.9)
evaluate_folder('evaluation/notdrone',.9)

evaluate_folder('evaluation/drone',.95)
evaluate_folder('evaluation/notdrone',.95)