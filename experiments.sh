#!/bin/bash

for i in 0 0.25 0.5 0.75
do
    for j in 0.25 0.75 1
    do
        echo "python main.py --shadow_dataset_name vggface2 --is_train_target True --is_train_shadow True --num_runs 10 --database_table_name all_retrain_runs --victim_ratio_shadow $i --victim_ratio $j"
    done
done
