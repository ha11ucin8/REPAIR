# Run REPAIR on ZsRE dataset (N=10 edits)
PYTHONPATH=$(pwd) python ./examples/run_wise_editing.py \
    --editing_method WISE \
    --hparams_dir hparams/WISE/llama3-8b.yaml \
    --data_dir data/wise \
    --data_type ZsRE \
    --ds_size 1000 --output_file ./outputs/llama3-8b-zsre-1000.json

PYTHONPATH=$(pwd) python ./examples/run_wise_editing.py \
    --editing_method WISE \
    --hparams_dir hparams/WISE/deepseek-r1-distill-qwen-1.5b.yaml \
    --data_dir data/wise \
    --data_type ZsRE \
    --ds_size 1000 --output_file ./outputs/deepseek-r1-distill-qwen-1.5b-zsre-1000.json

PYTHONPATH=$(pwd) python ./examples/run_wise_editing.py \
    --editing_method WISE \
    --hparams_dir hparams/WISE/qwen2.5-7b.yaml \
    --data_dir data/wise \
    --data_type ZsRE \
    --ds_size 1000 --output_file ./outputs/qwen2.5-7b-zsre-1000.json
