emb_modes=(5)
delimit_modes=(1)
train_size=1000003
test_size=1000003
nb_epoch=5
train_file="/mnt/data/strong_dns_augmented/urlnet/urlnet-train-1000000.0.txt"
test_file="/mnt/data/strong_dns_augmented/urlnet/urlnet-test-1000000.0.txt"
labelling="infected"

for ((i=0; i <${#emb_modes[@]}; ++i))
    do
    python train.py --data.data_dir ${train_file} \
    --data.dev_pct 0.001 --data.delimit_mode ${delimit_modes[$i]} --data.min_word_freq 1 \
    --model.emb_mode 5 --model.emb_dim 32 --model.filter_sizes 3,4,5,6 \
    --train.nb_epochs ${nb_epoch} --train.batch_size 1048 \
    --log.print_every 5 --log.eval_every 10 --log.checkpoint_every 10 \
    --log.output_dir runs/${train_size}_emb5_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/ --labelling ${labelling}

    python test.py --data.data_dir ${test_file} \
    --data.delimit_mode ${delimit_modes[$i]} \
    --data.word_dict_dir runs/${train_size}_emb5_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/words_dict.p \
    --data.subword_dict_dir runs/${train_size}_emb5_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/subwords_dict.p \
    --data.char_dict_dir runs/${train_size}_emb5_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/chars_dict.p \
    --log.checkpoint_dir runs/${train_size}_emb5_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/checkpoints/ \
    --log.output_dir runs/${train_size}_emb5_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/train_${train_size}_test_${test_size}.txt \
    --model.emb_mode 5 --model.emb_dim 32 --test.batch_size 1048 --labelling ${labelling}

    # python auc.py --input_path runs/${train_size}_emb5_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/ --input_file train_${train_size}_test_${test_size}.txt --threshold 0.5
    done
