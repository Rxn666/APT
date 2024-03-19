output_dir='result'
for seed in "3407" ; do
    python train.py --config-file config_k400_tiny.yaml \
        SEED ${seed} \
        OUTPUT_DIR "${output_dir}"
done