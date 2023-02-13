import os
import torch
import numpy as np
import parameters as p
import scipy.sparse as sp
import xc.method.mufin.model as lm
import xc.method.mufin.network as mn
import xc.libs.optimizer_utils as optimizer_utils

torch.backends.cudnn.enabled = False
# torch.multiprocessing.set_sharing_strategy('file_system')

__author__ = 'AM'


def train(model, params):
    model.fit(data_dir=params.data_dir, trn_img=params.trn_x_img,
              trn_txt=params.trn_x_txt, trn_lbl=params.trn_y,
              tst_img=params.tst_x_img, tst_txt=params.tst_x_txt,
              tst_lbl=params.tst_y, lbl_img=params.lbl_x_img,
              lbl_txt=params.lbl_x_txt)


def predict(model, params):
    if params.extract_y == "eye":
        params.extract_y = f"{params.num_labels}.eye"
    score_mat = model.predict(
        data_dir=params.data_dir, tst_img=params.extract_x_img,
        tst_txt=params.extract_x_txt, tst_lbl=params.extract_y,
        lbl_img=params.lbl_x_img, lbl_txt=params.lbl_x_txt)
    if isinstance(score_mat, dict):
        for key, val in score_mat.items():
            data_path = os.path.join(
                params.result_dir, f"{key}_{params.extract_fname}")
            if params.save_all:
                sp.save_npz(data_path, val, compressed=False)
        if not params.save_all:
            data_path = os.path.join(params.result_dir, params.extract_fname)
            sp.save_npz(data_path, val, compressed=False)
    else:
        val = score_mat
        print(val.shape)
        data_path = os.path.join(params.result_dir, params.extract_fname)
        sp.save_npz(data_path, val, compressed=False)


def predict_shorty(model, params):
    if params.extract_y == "eye":
        params.extract_y = f"{params.num_labels}.eye"
    score_mat = model.predict_shorty(
        data_dir=params.data_dir, tst_img=params.extract_x_img,
        tst_txt=params.extract_x_txt, tst_lbl=params.extract_y,
        tst_shorty=params.extract_x_shorty,
        lbl_img=params.lbl_x_img, lbl_txt=params.lbl_x_txt)
    if isinstance(score_mat, dict):
        for key, val in score_mat.items():
            data_path = os.path.join(
                params.result_dir, f"{key}_{params.extract_fname}")
            if params.save_all:
                sp.save_npz(data_path, val)
    else:
        val = score_mat
    if not params.save_all:
        data_path = os.path.join(params.result_dir, params.extract_fname)
        sp.save_npz(data_path, val)


def extract(model, params):
    embeddings = model.extract(data_dir=params.data_dir,
                               tst_img=params.extract_x_img,
                               tst_txt=params.extract_x_txt)
    out_path = os.path.join(params.result_dir, params.extract_fname)
    for key in embeddings.keys():
        embeddings[key].save(out_path+f".{key}")
    pass


def extract_model(model, params):
    encoder = model.extract_encoder()
    torch.save(encoder, os.path.join(params.result_dir, params.extract_fname))


def retrain_anns(model, params):
    model.retrain(data_dir=params.data_dir, trn_img=params.trn_x_img,
                  trn_txt=params.trn_x_txt, trn_lbl=params.trn_y,
                  lbl_img=params.lbl_x_img, lbl_txt=params.lbl_x_txt)


def construct_network(params):
    if params.module == 4:
        return getattr(mn, params.ranker)(params)
    return getattr(mn, params.model_fname)(params)


def construct_model(params, net, optimizer):
    if params.module == 4:
        return lm.MufinRanker(params, net, optimizer)
    return lm.Mufin(params, net, optimizer)


def main(params):
    """
        Main function
    """
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)
    network = construct_network(params)
    print("Model parameters: ", params)
    print(network)
    optimizer = None
    if params.mode == 'train':
        # NOTE: Use last index as padding label
        optimizer = optimizer_utils.Optimizer(optim=params.optim)
        model = construct_model(params, network, optimizer)
        train(model, params)

    elif params.mode == 'predict':
        # NOTE: Use last index as padding label
        model = construct_model(params, network, optimizer)
        predict(model, params)

    elif params.mode == 'predict_shorty':
        # NOTE: Use last index as padding label
        model = construct_model(params, network, optimizer)
        predict_shorty(model, params)

    elif params.mode == 'extract':
        # NOTE: Use last index as padding label
        model = construct_model(params, network, optimizer)
        extract(model, params)

    elif params.mode == 'extract_model':
        # NOTE: Use last index as padding label
        model = construct_model(params, network, optimizer)
        extract_model(model, params)

    elif params.mode == 'retrain_anns':
        # NOTE: Use last index as padding label
        model = construct_model(params, network, optimizer)
        retrain_anns(model, params)
    else:
        raise NotImplementedError("Unknown mode!")


if __name__ == '__main__':
    args = p.Parameters("Parameters")
    args.parse_args()
    main(args.params)
