# Instructions

## Get data
gsutil is required
```bash
mkdir ./data/binaries
gsutil -m cp gs://quickdraw_dataset/full/binary/*.bin ./data/binaries/
```

## Convert to TFRecords
To create TFRecord file per class:
```bash
cd code
python3 store_to_classes.py
```

## Train models
```bash
python3 {lstm,triplet-training,combined_model}.py
# Direct CNN model in ConvNet class of model_fns.py
```

## Test personal accuracy
[human.ipynb](human.ipynb) notebook can be used.

## KNN
[create_features.ipynb](create_features.ipynb) and [knn_visualization.ipynb](knn_visualization.ipynb) notebooks are used for representation storage, to get nearest neighbors and apply KNN classification.