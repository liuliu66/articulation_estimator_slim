# Articulation Estimator Slim

Articulation object pose estimation and joint prediction

## Installation

```
$ pip install -r requirements.txt
```

## usage

# train

```
$ python train.py
```

please refer to the dataset structure in 'data' folder.

# inference

```
$ python test.py checkpoints/checkpoint.pth demo demo/det_bboxes.json
```

the inferred image is outputted in demo/output. 
You can change the test_id in test.py to infer other image.
