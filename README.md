# [Multimodal extreme classification](https://github.com/Extreme-classification/MUFIN/blob/main/MUFIN.pdf)
```bib
@InProceedings{Mittal22, 
    author    = {Mittal, A. and Dahiya, K. and Malani, S. and Ramaswamy, J. and Kuruvilla, S. and 
                 Ajmera, J. and Chang, K. and Agrawal, S. and Kar, P. and Varma, M.},     
    title     = {MUFIN: Multimodal extreme classification},
    booktitle = {CVPR}, 
    month     = {June},
    year      = {2022}
}
```

## SETUP WORKSPACE
```bash
mkdir -p ${HOME}/scratch/XC/data 
mkdir -p ${HOME}/scratch/XC/programs
```

## SETUP DATASET
Download dataset from [XML Repository](http://manikvarma.org/downloads/XC/XMLRepository.html)
```bash
cd ${HOME}/scratch/XC/data
gdown --id <from google form>
unzip MM-AmazonTitles-300K.zip
cd -
```

## SETUP MUFIN
```bash
cd ${HOME}/scratch/XC/programs
git clone https://github.com/anshumitts/CafeXC.git
conda env create -f CafeXC/cafexc.yml
conda activate xc

pip install Cython git+https://github.com/kunaldahiya/pyxclib.git

git clone https://github.com/Extreme-classification/MUFIN.git
```

## RUNNING MUFIN
```bash
cd ${HOME}/scratch/XC/programs/MUFIN
chmod +x run_MUFIN.sh
./run_MUFIN.sh <ALL_GPU_IDS> <TYPE> <DATASET> <FOLDER_NAME> <IMG_ENCODER> <TXT_ENCODER> <KEEP_TOP_K> <RESTRICTMEM>
# TYPE          :	MufinMultiModal PreTrainedMufinMultiModal
# DATASET       :	MM-AmazonTitles-300K
# FOLDER_NAME   :	USER's choice
# IMG_ENCODER   :	ViT resnet18 vgg11 resnet50FPN
# TXT_ENCODER   :	sentencebert BoW Seq VisualBert 
# KEEP_TOP_K    :   USE ONLY K images [-1, inf]; -1 will use all images
# RESTRICTMEM   :   0 will load all data in RAM while 1 will load from disk.
e.g.
./run_MUFIN.sh 0,1 PreTrainedMufinMultiModal MM-AmazonTitles-300K MUFIN_pretrained ViT sentencebert -1 0
./run_MUFIN.sh 0,1 MufinMultiModal MM-AmazonTitles-300K MUFIN ViT sentencebert -1 0
```
