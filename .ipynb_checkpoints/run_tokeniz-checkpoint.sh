export work_dir="${work_dir}"
export PROGRAMS_DIR="${PROGRAMS_DIR}"

dataset=${1}     # SAMPLE-IMG-AmazonTitles-1K
txt_model="${2}" # BoW Seq VisualBert sentencebert
data_dir="${work_dir}/data"
temp_model_data="img-xml_data"
dset_json="${dataset}.json"
meta_data_folder="${data_dir}/$dataset/${temp_model_data}/$txt_model"

mkdir -p "${data_dir}/${dataset}/${txt_model}" "${meta_data_folder}"

get_obj() {
    echo $(python3 -c "import json; print(json.load(open('configs/$dset_json'))['DEFAULT']['$1'])")
}

tokenize_text() {
    raw_dset=$(get_obj raw_data_name)
    max_len=$(get_obj max_len)
    raw_dir="${work_dir}/RawData/${raw_dset}"
    num_vocab=$(get_obj num_vocab)
    out_dir="${meta_data_folder}"
    trn_ft_file="${data_dir}/${dataset}/$txt_model/trn_X_Xf.seq.memmap"
    tst_ft_file="${data_dir}/${dataset}/$txt_model/tst_X_Xf.seq.memmap"
    zsh_ft_file="${data_dir}/${dataset}/$txt_model/zsh_X_Xf.seq.memmap"
    lbl_ft_file="${data_dir}/${dataset}/$txt_model/lbl_X_Xf.seq.memmap"
    trn_lb_file="${data_dir}/${dataset}/trn_X_Y.txt"
    mkdir -p ${out_dir}/tokenizer
    args="--data_dir ${data_dir} --raw_dir ${raw_dir} \
    --out_dir $out_dir --txt_model $txt_model --n_vocab ${num_vocab}\
    --trn_map ${raw_dir}/train_map.txt --tst_map ${raw_dir}/test_map.txt \
    --lbl_map ${raw_dir}/label_map.txt --zsh_map ${raw_dir}/zsh_map.txt --max_len $max_len \
    --trn_xf $trn_ft_file --tst_xf $tst_ft_file --lbl_xf $lbl_ft_file --zsh_xf $zsh_ft_file \
    --trn_y ${trn_lb_file}"
    python ${PROGRAMS_DIR}/xc/tools/tokenize_text.py $args
}

if [ ! -e "${meta_data_folder}/features_split.txt" ]; then
    tokenize_text
fi
