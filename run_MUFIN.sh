# ./run_MUFIN.sh 0,1 PreTrainedMufinMultiModal MM-AmazonTitles-300K MUFIN ViT sentencebert -1 0
# ./run_MUFIN.sh 0,1 MufinMultiModal MM-AmazonTitles-300K TrainingMUFIN ViT sentencebert 5 0
export work_dir="${HOME}/scratch/XC"
export PROGRAMS_DIR="${work_dir}/programs/CafeXC"
export PYTHONPATH="${PYTHONPATH}:${PROGRAMS_DIR}"
export CUDA_VISIBLE_DEVICES=${1} # CUDA devices
model_type=${2}                  # MultiModalSiameseXC SiameseTextXML
dataset=${3}                     # SAMPLE-IMG-AmazonTitles-1K
version=${4}                     # ANY_NAME
img_model="${5}"                 # resnet18 vgg11 inception_v3 ViT
txt_model="${6}"                 # BoW Seq VisualBert sentencebert
export KEEP_TOP_K="${7}"         # Keeps only first two images
export RESTRICTMEM="${8}"        # 0-> stores on ram, 1-> stores on disk

data_dir="${work_dir}/data"
model_dir="${work_dir}/models/${dataset}/${model_type}/v_${version}"
result_dir="${work_dir}/results/${dataset}/${model_type}/v_${version}"
temp_model_data="img-xml_data"
dset_json="${dataset}.json"
meta_data_folder="${data_dir}/$dataset/${temp_model_data}/$txt_model"
validate_args="--validate"

PARAMS="--model_fname ${model_type} \
--img_model ${img_model} --data_dir ${data_dir}/${dataset} \
--txt_model $txt_model --config configs/${dset_json} \
--model_dir ${model_dir} --result_dir ${result_dir} \
--seed 22 --dataset $dataset --pred_fname score \
--filter_labels filter_labels_test.txt \
${validate_args} --doc_first"

extension="bin"
suffix="images"
if [[ "${model_type}" == *"PreTrained"* ]]; then
    extension="vect"
    suffix="pre-trained/${img_model}"
fi

mkdir -p $result_dir $model_dir
mkdir -p ${result_dir}/module1
mkdir -p ${result_dir}/module2
mkdir -p ${result_dir}/module3
mkdir -p ${result_dir}/module4

run_eval() {
    log_eval_file="$result_dir/log_eval.txt"
    python -u ${PROGRAMS_DIR}/xc/tools/evaluate.py \
        "$data_dir/$dataset/trn_X_Y.txt" \
        "$data_dir/$dataset/${2}" \
        "$result_dir" "${1}" "${data_dir}/$dataset" \
        "configs/${dset_json}" \
        "${3}" "${4}" 2>&1 | tee -a $log_eval_file
}

module1() {
    log_tr_file="${result_dir}/log_train.txt"
    python -W ignore -u mufin.py $PARAMS --mode train --module 1 | tee $log_tr_file
    python -W ignore -u mufin.py $PARAMS --mode retrain_anns --module 1 | tee $log_pr_file
}

module2() {
    log_pr_file="${result_dir}/log_predict.txt"
    extra_args="--extract_x_img images/test.img.bin --extract_x_txt raw_data/test.raw.txt \
    --extract_fname module2/test.npz --extract_y tst_X_Y.txt --filter_labels filter_labels_test.txt"
    python -u mufin.py $PARAMS --mode predict ${extra_args} --module 2 | tee $log_pr_file
    run_eval "module2/test" "tst_X_Y.txt" "filter_labels_test.txt" "m2"

    extra_args="--extract_x_img images/train.img.bin --extract_x_txt raw_data/train.raw.txt \
    --extract_fname module2/train.npz --extract_y trn_X_Y.txt --filter_labels filter_labels_train.txt"
    python -u mufin.py $PARAMS --mode predict ${extra_args} --module 2 | tee -a $log_pr_file
    run_eval "module2/train" "trn_X_Y.txt" "filter_labels_train.txt" "m2"
}

module3() {
    log_ex_file="${result_dir}/log_extract.txt"
    extra_args="--extract_fname module3/encoder.pkl"
    python -u mufin.py $PARAMS --mode extract_model --module 3 ${extra_args} | tee $log_ex_file

    extra_args="--extract_x_img images/test.img.bin \
    --extract_x_txt raw_data/test.raw.txt \
    --extract_fname module3/test"
    python -u mufin.py $PARAMS --mode extract ${extra_args} --module 3 | tee -a $log_ex_file

    extra_args="--extract_x_img images/train.img.bin \
    --extract_x_txt raw_data/train.raw.txt \
    --extract_fname module3/train"
    python -u mufin.py $PARAMS --mode extract ${extra_args} --module 3 | tee -a $log_ex_file

    extra_args="--extract_x_img images/label.img.bin \
    --extract_x_txt raw_data/label.raw.txt \
    --extract_fname module3/label"
    python -u mufin.py $PARAMS --mode extract ${extra_args} --module 3 | tee -a $log_ex_file
}

module4() {
    ranker=$1
    log_rk_file="${result_dir}/log_ranker_${ranker}.txt"
    XPARAMS="${PARAMS} --model_out_name model_${ranker}.pkl --ranker ${ranker} \
    --extract_x_img module3/test.img.pretrained  --extract_x_txt module3/test.txt.pretrained \
    --extract_y tst_X_Y.txt --extract_fname test_${ranker} --module 4 --save_all ${validate_args}"
    python -u mufin.py $XPARAMS --mode train | tee $log_rk_file
    python -u mufin.py $XPARAMS --mode predict | tee -a $log_rk_file".pred"
    run_eval "test_${ranker}" "tst_X_Y.txt" "filter_labels_test.txt" "m4"
}

module4pp() {
    ranker=$1
    extension="bin"
    log_rk_file="${result_dir}/log_ranker_${ranker}.txt"
    XPARAMS="${PARAMS} --model_out_name model_${ranker}.pkl --ranker ${ranker} \
    --extract_x_img images/test.img.bin --extract_x_txt raw_data/test.raw.txt \
    --extract_y tst_X_Y.txt --extract_fname test_${ranker} --module 4 --save_all \
    --filter_labels filter_labels_test.txt --encoder_init module3/encoder.pkl ${validate_args}"
    python -u mufin.py $XPARAMS --mode train | tee $log_rk_file
    python -u mufin.py $XPARAMS --mode predict | tee -a $log_rk_file".pred"
    run_eval "test_${ranker}" "tst_X_Y.txt" "filter_labels_test.txt" "m4"
}

fetch_scoremat() {
    ranker=$1
    alpha=$2
    log_eval_file="$result_dir/log_extract.txt"
    echo "${ranker}" | tee -a $log_eval_file
    python -u ${PROGRAMS_DIR}/xc/tools/extract_eval.py "$data_dir/$dataset/trn_X_Y.txt" \
        "$data_dir/$dataset/tst_X_Y.txt" "$result_dir" "test_${ranker}" "${data_dir}/$dataset" \
        "configs/${dset_json}" "filter_labels_test.txt" "m4" $alpha 2>&1 | tee -a $log_eval_file
}

MUFIN() {
    ranker=$1
    rm -rf ${model_dir}/mufin_${ranker}
    rm -rf ${result_dir}/mufin_${ranker}

    mkdir -p ${model_dir}/mufin_${ranker}
    mkdir -p ${result_dir}/mufin_${ranker}
    mkdir -p ${result_dir}/mufin_${ranker}/module2
    mkdir -p ${result_dir}/mufin_${ranker}/module4

    cp ${model_dir}/model.pkl ${model_dir}/mufin_${ranker}
    cp ${model_dir}/filter_model.pkl ${model_dir}/mufin_${ranker}

    cp ${model_dir}/model_${ranker}.pkl ${model_dir}/mufin_${ranker}
    cp ${model_dir}/filter_model_${ranker}.pkl ${model_dir}/mufin_${ranker}
    model_dir="${model_dir}/mufin_${ranker}"
    result_dir="${result_dir}/mufin_${ranker}"

    PARAMS="--model_fname ${model_type} \
            --img_model ${img_model} --data_dir ${data_dir}/${dataset} \
            --txt_model $txt_model --config configs/${dset_json} \
            --model_dir ${model_dir} --result_dir ${result_dir} \
            --seed 22 --dataset $dataset --pred_fname score \
            --filter_labels filter_labels_test.txt ${validate_args}"

    log_ex_file="${result_dir}/log_extract_mufin.txt"
    extra_args="${PARAMS} --model_out_name model_${ranker}.pkl \
    --extract_fname encoder.pkl --ranker ${ranker}"
    python -u mufin.py --mode extract_model --module 4 ${extra_args} | tee $log_ex_file

    log_pr_file="${result_dir}/log_mufin.txt"
    python -u mufin.py $PARAMS --mode retrain_anns \
        --encoder_init encoder.pkl --module 1 | tee $log_pr_file

    extra_args="$PARAMS --extract_x_img images/test.img.bin --extract_x_txt raw_data/test.raw.txt \
    --extract_fname module2/test.npz --extract_y tst_X_Y.txt --filter_labels filter_labels_test.txt"
    python -u mufin.py --mode predict ${extra_args} --module 2 | tee -a $log_pr_file
    run_eval "module2/test" "tst_X_Y.txt" "filter_labels_test.txt" "m2"

    extra_args="${PARAMS} --model_out_name model_${ranker}.pkl --ranker ${ranker} \
    --extract_x_img images/test.img.bin --extract_x_txt raw_data/test.raw.txt \
    --extract_y tst_X_Y.txt --extract_fname test_m4_mufin_${ranker} --save_all \
    --filter_labels filter_labels_test.txt ${validate_args}"
    python -u mufin.py --mode predict ${extra_args} --module 4 | tee -a $log_rk_file".pred"
    run_eval "test_m4_mufin_${ranker}" "tst_X_Y.txt" "filter_labels_test.txt" "m4"
}

module1
module2
module3

module4 MufinXAttnRanker
# module4pp MufinXAttnRankerpp
# fetch_scoremat MufinXAttnRankerpp 1
# fetch_scoremat MufinXAttnRanker 0.5
# MUFIN MufinXAttnRankerpp
