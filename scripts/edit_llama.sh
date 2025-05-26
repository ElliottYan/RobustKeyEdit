HF_ENDPOINT=https://hf-mirror.com
ROOT=path_to_your_dir/EasyEditOurs
cd $ROOT


for seed in 0; do
    RESULT_DIR=$ROOT/results/llama2_7b
    mkdir -p $RESULT_DIR
    mkdir -p $RESULT_DIR/metrics
    mkdir -p $RESULT_DIR/models

    export CUDA_VISIBLE_DEVICES=$2

    editing_method=$1

    # baseline
    python $ROOT/edit_single.py \
        --editing_method=$editing_method \
        --hparams_dir=our_hparams/$editing_method/llama-7b.yaml \
        --data_dir $ROOT/../data/test.zsre_para_prompt_para_subject_llama2-7b.json \
        --metrics_save_dir $RESULT_DIR/metrics \
        --model_save_dir $RESULT_DIR/models/$editing_method \
        --update_hparams '{"mom2_adjustment": "True", "mom2_dataset": "wikitext", "use_our_aggregator": ""}' \
        --collect 'original_shuffle_long_rephrase' \
        --seed $seed
    
    

     python $ROOT/analysis_src/compute_single_edit_metrics_final.py $RESULT_DIR/metrics/${editing_method}_results_seed$seed.json

    for tr in 0.9; do
        for target_consistency_weight in 50000; do
            RESULT_DIR=$ROOT/results/llama2_7b/rep_tr${tr}_${target_consistency_weight}
            mkdir -p $RESULT_DIR
            mkdir -p $RESULT_DIR/metrics
            mkdir -p $RESULT_DIR/models
            
            editing_method=$1
            python $ROOT/edit_single.py \
                --editing_method=$editing_method \
                --hparams_dir=our_hparams/$editing_method/llama-7b.yaml \
                --data_dir $ROOT/../data/test.zsre_para_prompt_para_subject_llama2-7b.json \
                --metrics_save_dir $RESULT_DIR/metrics \
                --model_save_dir $RESULT_DIR/models/$editing_method \
                --update_hparams '{"mom2_adjustment": "True", "mom2_dataset": "wikitext", "use_our_aggregator": "True", "agg_loss_weight": 1.0, "kl_loss_weight":0.0, "spread_loss_weight": 0.0, "agg_add_long_context": "True", "agg_add_shuffle":"True", "agg_remove_c": "True", "num_train_agg_steps": 10, "agg_use_cls": "True", "agg_train_on_held": "True", "cls_threshold": '$tr', "target_consistency_weight": '$target_consistency_weight'}' \
                --collect 'original_shuffle_long_rephrase' \
                --seed $seed
                
            echo ${tr}_${target_consistency_weight}
            python $ROOT/analysis_src/compute_single_edit_metrics_final.py $RESULT_DIR/metrics/${editing_method}_results_seed$seed.json
        done
    done
done