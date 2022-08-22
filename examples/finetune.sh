DATASET=$1
DESIGN_SN=$2
GRIDSEARCH=1
thread_num=1
count=0
log_prefix="log/TB"
log_dir="$log_prefix-${DATASET}/iter-$DESIGN_SN"
if [ -d "model_dir/${DATASET}/${DESIGN_SN}" ]; then
	rm -r "model_dir/${DATASET}/${DESIGN_SN}"
fi
mkdir -p "model_dir/${DATASET}/${DESIGN_SN}"
rm -r "data/${DATASET}/processed"
if [ "${GRIDSEARCH}" == "1" ]; then
    lrs_list="1e-3 5e-5 1e-4 5e-4"
    model_list="Tankbind Halfbind init"
    if [ "${DATASET}" == "PRMT5" ]; then
        batch_size_list="32 48"
    else
        batch_size_list="16 32 48"
    fi
else
    # lrs_list="1e-3"
    # model_list="Tankbind"
    # batch_size_list="32"
    lrs_list="1e-3"
    model_list="Tankbind"
    batch_size_list="32"
fi
dropout_rate=0
for model_mode in $model_list; do
    for lr in $lrs_list; do
        for batch_size in $batch_size_list; do
            log_file="$log_dir/lr${lr}-batch${batch_size}-${model_mode}.txt"
            echo "Outputs redirected to $log_file"
            mkdir -p $log_dir
            for time in $(seq 1 1); do
                {
                    python finetune.py \
                            --batch_size=$batch_size \
                            --max_epoch=25 \
                            --iter=${DESIGN_SN} \
                            --compound_name=${DATASET} \
                            --init_model="../saved_models/self_dock.pt" \
                            --lr=$lr \
                            --model_mode=$model_mode \
                            --dropout_rate=$dropout_rate >> $log_file 2>&1
                    cat $log_dir/* | grep 'TEST' > $log_dir/final_result
                } &
                let count+=1
                if [[ $(($count % $thread_num)) -eq 0 ]]; then
                    wait
                fi
            done
        done    
    done
done
