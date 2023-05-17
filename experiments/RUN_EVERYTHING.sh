#!/bin/bash

for partition in baseline corrects; do
    bash experiments/rc_train_exp.sh $partition
    bash experiments/rc_train_sigmorphon_resampled_exp.sh $partition
    bash experiments/rc_train_sig_res_reinflection_exp.sh $partition
done