# ./run_pretrained.sh ViT
model="$1"
work_dir="${HOME}/scratch/XC"
export PROGRAMS_DIR="${work_dir}/programs/ExtremeMethods"
export PYTHONPATH="${PYTHONPATH}:${PROGRAMS_DIR}"

dset_name="MM-AmazonTitles-300K"
data_dir="${work_dir}/data/${dset_name}"

fetch_pre_trained() {
    echo "Generation pre-trained features"
    out_dir="${data_dir}/$model"
    rm -rf $out_dir
    echo $out_dir
    mkdir -p $out_dir
    if [ -e "${data_dir}/images/test.img.bin.npz" ]; then
        python -u $PROGRAMS_DIR/xc/tools/get_pre_train.py --data_dir $data_dir --output_dir $out_dir \
            --model $model --mode "test" --output_file "test"
    fi
    if [ -e "${data_dir}/images/train.img.bin.npz" ]; then
        python -u $PROGRAMS_DIR/xc/tools/get_pre_train.py --data_dir $data_dir --output_dir $out_dir \
            --model $model --mode "train" --output_file "train"
    fi
    if [ -e "${data_dir}/images/label.img.bin.npz" ]; then
        python -u $PROGRAMS_DIR/xc/tools/get_pre_train.py --data_dir $data_dir --output_dir $out_dir \
            --model $model --mode "label" --output_file "label"
    fi
}

fetch_pre_trained
