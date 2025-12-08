our code is based on https://github.com/ziyaow1010/FederatedLLM/

run !python main.py \
    --global_model #yourmodel \
    --data_path "./data_wiz/10" \ # reuse data in ziyaow1010， not support another dataset now
    --output_dir "./runs/fedit_test" \
    --num_clients 10 \ # max is 10
    --num_communication_rounds 7 \
    --local_num_epochs 1 \
    --local_batch_size 128 \
    --local_micro_batch_size 16 \
    --cutoff_len 512 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --freezeA_after_rounds 2  # when =-1 never freeze, when =0 freeze at beginning,
