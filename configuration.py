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
conf.image_size = (900, 1256)  # (h, w)

conf.class_nums = {
    'KLAC': 951,
    'KLFE': 1766,
    'KLHC': 1069
}
conf.label_name = {
    'KLAC': 'sp_sfb',
    'KLFE': 'sp_gg',
    'KLHC': 'sp_qn'
}

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

# Deployment configuration
# ==============================================================================================
conf.deployment_save_dir = 'deploy'

# TESTSET CORRECTION
# ---------------------------------------------------------------------------------
KLFE_CORRECTION = ['KLFE1089_37.jpg', 'KLFE1096_49.jpg', 'KLFE1101_71.jpg', 'KLFE1108_65.jpg', 'KLFE1114_36.jpg',
                   'KLFE1124_77.jpg', 'KLFE1127_58.jpg', 'KLFE1133_54.jpg', 'KLFE1134_39.jpg', 'KLFE1135_103.jpg',
                   'KLFE1143_55.jpg', 'KLFE1144_80.jpg', 'KLFE1148_59.jpg', 'KLFE1149_74.jpg', 'KLFE1152_46.jpg',
                   'KLFE1155_76.jpg', 'KLFE1159_83.jpg', 'KLFE1161_59.jpg', 'KLFE1167_53.jpg', 'KLFE1186_45.jpg',
                   'KLFE1220_32.jpg', 'KLFE1229_45.jpg', 'KLFE1232_54.jpg', 'KLFE1263_97.jpg', 'KLFE1303_31.jpg',
                   'KLFE1305_49.jpg', 'KLFE1321_79.jpg', 'KLFE1382_50.jpg', 'KLFE1457_89.jpg', 'KLFE1471_57.jpg',
                   'KLFE1518_66.jpg', 'KLFE1525_35.jpg', 'KLFE1551_60.jpg', 'KLFE1608_77.jpg', 'KLFE1616_65.jpg']

KLAC_CORRECTION = ['KLAC0477_42.jpg', 'KLAC0480_87.jpg', 'KLAC0506_87.jpg', 'KLAC0508_71.jpg', 'KLAC0509_69.jpg',
                   'KLAC0510_85.jpg', 'KLAC0523_31.jpg', 'KLAC0532_64.jpg', 'KLAC0556_58.jpg', 'KLAC0582_39.jpg',
                   'KLAC0589_48.jpg', 'KLAC0630_54.jpg', 'KLAC0657_94.jpg', 'KLAC0743_63.jpg', 'KLAC0745_65.jpg',
                   'KLAC0746_69.jpg', 'KLAC0757_43.jpg', 'KLAC0771_68.jpg', 'KLAC0779_37.jpg', 'KLAC0780_88.jpg',
                   'KLAC0790_43.jpg', 'KLAC0814_122.jpg', 'KLAC0832_84.jpg', 'KLAC0842_32.jpg', 'KLAC0860_81.jpg',
                   'KLAC0861_68.jpg', 'KLAC0863_67.jpg', 'KLAC0864_82.jpg', 'KLAC0867_37.jpg', 'KLAC0870_74.jpg',
                   'KLAC0876_39.jpg', 'KLAC0878_60.jpg', 'KLAC0885_43.jpg', 'KLAC0905_53.jpg', 'KLAC0912_67.jpg',
                   'KLAC0923_79.jpg', 'KLAC0935_42.jpg', 'KLAC0940_76.jpg', 'KLAC0945_87.jpg', 'KLAC0946_44.jpg']

KLHC_CORRECTION = ['KLHC0551_63.jpg', 'KLHC0555_96.jpg', 'KLHC0559_46.jpg', 'KLHC0560_1.jpg', 'KLHC0587_57.jpg',
                   'KLHC0590_72.jpg', 'KLHC0616_71.jpg', 'KLHC0619_83.jpg', 'KLHC0670_88.jpg', 'KLHC0671_77.jpg',
                   'KLHC0683_70.jpg', 'KLHC0693_4.jpg', 'KLHC0729_53.jpg', 'KLHC0766_101.jpg', 'KLHC0768_99.jpg',
                   'KLHC0772_68.jpg', 'KLHC0775_50.jpg', 'KLHC0776_69.jpg', 'KLHC0777_99.jpg', 'KLHC0778_137.jpg',
                   'KLHC0779_56.jpg', 'KLHC0780_114.jpg', 'KLHC0783_80.jpg', 'KLHC0791_138.jpg', 'KLHC0793_77.jpg',
                   'KLHC0794_83.jpg', 'KLHC0804_60.jpg', 'KLHC0829_111.jpg', 'KLHC0832_82.jpg', 'KLHC0835_99.jpg',
                   'KLHC0837_67.jpg', 'KLHC0864_7.jpg', 'KLHC0866_37.jpg', 'KLHC0867_24.jpg', 'KLHC0871_109.jpg',
                   'KLHC0872_103.jpg', 'KLHC0874_86.jpg', 'KLHC0883_101.jpg', 'KLHC0886_98.jpg', 'KLHC0916_72.jpg',
                   'KLHC0918_98.jpg', 'KLHC0939_78.jpg', 'KLHC0940_49.jpg', 'KLHC1002_129.jpg', 'KLHC1009_84.jpg',
                   'KLHC1010_82.jpg', 'KLHC1030_91.jpg', 'KLHC1065_67.jpg', 'KLHC1069_49.jpg']
