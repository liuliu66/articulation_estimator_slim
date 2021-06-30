from .articulation_dataset import ArticulationDataset
from .pipelines_train import *
from .pipelines_test import *

train_keys = ['parts_pts', 'parts_pts_feature', 'parts_cls', 'mask_array',
                    'nocs_p', 'nocs_g', 'offset_heatmap',
                    'offset_unitvec', 'joint_orient', 'joint_cls',
                    'joint_cls_mask', 'joint_params']
train_pipelines = [CreatePointDataTrain(),
                   LoadArtiNOCSData(),
                   LoadArtiJointData(),
                   CreateArtiJointGT(),
                   DownSampleTrain(),
                   ToTensor(keys=train_keys),
                   Collect(keys=train_keys,
                         meta_keys=['img_prefix', 'sample_name', 'norm_factors', 'corner_pts',
                                    'joint_ins'])]

test_keys = ['pts', 'pts_feature']
test_pipelines = [CreatePointData(), DownSample(), ToTensor(keys=test_keys)]