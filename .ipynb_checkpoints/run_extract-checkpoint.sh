export work_dir="${work_dir}"
export PROGRAMS_DIR=${PROGRAMS_DIR}
export PYTHONPATH="${PYTHONPATH}"

model_type=${1} # MultiModalSiameseXC SiameseTextXML
dataset=${2}    # SAMPLE-IMG-AmazonTitles-1K
version=${3}    # ANY_NAME
ranker="${4}"   # XAttnRanker-hybrid XAttnRanker NGAME
alpha="${5}"    # alpha = M4 weight and 1-alpha = M2 weight
beta="${6}"     # 1-beta = Fusion score weight

data_dir="${work_dir}/data"
result_dir="${work_dir}/results/${dataset}/${model_type}/v_${version}"
dset_json="${dataset}.json"

run_eval() {
    log_eval_file="$result_dir/log_extract.txt"
    echo "${ranker}" | tee -a $log_eval_file
    python -u ${PROGRAMS_DIR}/xc/tools/extract_eval.py \
        "$data_dir/$dataset/trn_X_Y.txt" \
        "$data_dir/$dataset/${2}" \
        "$result_dir" "${1}" "${data_dir}/$dataset" \
        "configs/${dset_json}" \
        "${3}" "${4}" $alpha $beta 2>&1 | tee -a $log_eval_file
}

run_eval "test_${ranker}" "tst_X_Y.txt" "filter_labels_test.txt" "m4"
