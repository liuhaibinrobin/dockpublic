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
if [ "${GRIDSEARCH}" == "1" ]; then
    lrs_list="1e-3 5e-5 1e-4 5e-4 5e-3"
    drop_list="0 0.1"
else
    lrs_list="1e-3"
    drop_list="0.1"
fi
for lr in $lrs_list; do
    for dropout_rate in $drop_list; do
        log_file="$log_dir/lr${lr}-drop${dropout_rate}.txt"
        echo "Outputs redirected to $log_file"
        mkdir -p $log_dir
        for time in $(seq 1 1); do
            {
                python finetune.py \
                        --batch_size=32 \
                        --max_epoch=22 \
                        --iter=${DESIGN_SN} \
                        --compound_name=${DATASET} \
                        --init_model="../saved_models/self_dock.pt" \
                        --lr=$lr \
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
