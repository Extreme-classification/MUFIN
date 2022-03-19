# ./setup_dataset.sh 0,1,2,3 $HOME/XC D2Q D2Q D2Q ViT 1 1 1 
model="$1"
work_dir="${HOME}/scratch/XC"
export PROGRAMS_DIR="${work_dir}/programs/ExtremeMethods"
export PYTHONPATH="${PYTHONPATH}:${PROGRAMS_DIR}"

dset_name="MM-AmazonTitles-300K"
raw_dset_name="MM-Amazon-300K"
data_dir="${work_dir}/data/${dset_name}"
raw_dir="${work_dir}/RawData/${raw_dset_name}"
pre_trained_dir="${work_dir}/RawData/${raw_dset_name}/pre-trained"

mkdir -p "${pre_trained_dir}"

fetch_pre_trained() {
    echo "Generation pre-trained features"
    out_dir="${pre_trained_dir}/$model"
    rm -rf $out_dir
    echo $out_dir
    mkdir -p $out_dir
    if [ -e "${raw_dir}/images/test.img.bin.npz" ]; then
        python -u $PROGRAMS_DIR/xc/tools/get_pre_train.py --data_dir $raw_dir --output_dir $out_dir \
            --model $model --mode "test" --output_file "test"
    fi
    if [ -e "${raw_dir}/images/train.img.bin.npz" ]; then
        python -u $PROGRAMS_DIR/xc/tools/get_pre_train.py --data_dir $raw_dir --output_dir $out_dir \
            --model $model --mode "train" --output_file "train"
    fi
    if [ -e "${raw_dir}/images/label.img.bin.npz" ]; then
        python -u $PROGRAMS_DIR/xc/tools/get_pre_train.py --data_dir $raw_dir --output_dir $out_dir \
            --model $model --mode "label" --output_file "label"
    fi
    
    if [ -e "${raw_dir}/images/zsh.img.bin.npz" ]; then
        python -u $PROGRAMS_DIR/xc/tools/get_pre_train.py --data_dir $raw_dir --output_dir $out_dir \
            --model $model --mode "zsh" --output_file "zsh"
    fi
    cd -
}

fetch_pre_trained