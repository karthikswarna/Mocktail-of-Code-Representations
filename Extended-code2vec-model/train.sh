#!/usr/bin/env bash
###########################################################
# Change the following values to train a new model.
# type: the name of the new model, only affects the saved file name.
# dataset: the name of the dataset, as was preprocessed using preprocess.sh
# test_data: by default, points to the validation set, since this is the set that
#   will be evaluated after each training iteration. If you wish to test
#   on the final (held-out) test set, change 'val' to 'test'.
type=ast_cfg_ddg
dataset_name=testdata
data_dir=data/${dataset_name}/${type}
data=${data_dir}/${dataset_name}_${type}
val_data=${data_dir}/${dataset_name}_${type}.val.c2v
test_data=${data_dir}/${dataset_name}_${type}.test.c2v
model_dir=models/${dataset_name}/${type}

mkdir -p ${model_dir}
set -e
## Training and evaluating the model on training and validation data.
python3 -u run.py --reps ast cfg ddg --max_contexts '{"ast":"200", "cfg":"10", "ddg":"100"}' --data ${data} --test ${val_data} --save ${model_dir}/saved_model

## Evaluate a trained model on test data (by loading the model)
# python3 -u run.py --reps ast cfg ddg --max_contexts '{"ast":"200", "cfg":"10", "ddg":"100"}' --load ${model_dir}/saved_model --test ${test_data}
