output_dir='experiment_result'
for seed in "3407" ; do
    python train.py --config-file config_hmdb.yaml \
        SEED ${seed} \
        OUTPUT_DIR "${output_dir}"
done