import os
from easydict import EasyDict as edict

conf = edict()

conf.project_root = os.path.join(os.environ['HOME'], 'Lab/tensorflow/retinanet-tensorflow')

# Dataset configuration
# ==============================================================================================
conf.dataset_root = os.path.join(conf.project_root, 'data')
conf.image_dir = os.path.join(conf.dataset_root, 'VOC2007/JPEGImages')
conf.annot_dir = os.path.join(conf.dataset_root, 'VOC2007/Annotations')
conf.split_dir = os.path.join(conf.dataset_root, 'VOC2007/ImageSets/Main')
conf.class_name = (
    'sp_sfb', 'sp_dn', 'sp_cns', 'sp_qn', 'sp_xn', 'sp_gg',
    'nsp_sfb', 'nsp_dn', 'nsp_cns', 'nsp_qn', 'nsp_xn', 'nsp_gg')
# NOTE: range starts from 1
conf.num_class = len(conf.class_name)
conf.name_to_label_map = {v: k for k, v in enumerate(conf.class_name, start=1)}

# Model configuration
# ==============================================================================================
# nms score threshold
conf.nms_thred = 0.5
# score threshold
conf.cls_thred = 0.5
# max detections
conf.max_output_size = 10

# Training configuration
# ==============================================================================================
conf.checkpoint_dir = os.path.join(conf.project_root, 'checkpoints')
conf.summary_dir = os.path.join(conf.project_root, 'summary')
conf.log_interval = 10

conf.input_size = (448, 448)
conf.batch_size = 16
conf.num_epochs = 50
conf.buffer_size = 2000

conf.shuffle = True
conf.learning_rate = 1e-5

