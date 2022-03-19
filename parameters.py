from xc.libs.parameters import ParameterBase, nullORstr


class Parameters(ParameterBase):

    def _construct(self):

        self.parser.add_argument('--data_dir', action='store', type=str,
                                 help='path to main data directory')

        self.parser.add_argument('--dataset', type=str,
                                 help='dataset name')

        self.parser.add_argument('--model_dir', type=str,
                                 help='directory to store models')

        self.parser.add_argument('--result_dir', type=str,
                                 help='directory to store results')

        self.parser.add_argument('--emb_dir', type=str, default="random",
                                 help='directory of word embeddings')

        self.parser.add_argument('--model_fname', default='model',
                                 type=str, help='model file name')

        self.parser.add_argument('--model_out_name', default='model.pkl',
                                 type=str, help='model file name')

        self.parser.add_argument('--img_model', default='vgg11', type=str,
                                 help='model file name')

        self.parser.add_argument('--txt_model', default='BoW', type=str,
                                 help='text encoder file name')

        self.parser.add_argument('--config', type=str,
                                 help='model config files')

        self.parser.add_argument('--mode', default='predict', type=str,
                                 help='model mode')

        self.parser.add_argument('--seed', default=22, type=int,
                                 help='model instance id')

        self.parser.add_argument('--pred_fname', default="test_predictions",
                                 type=str, help='prediction fname')

        self.parser.add_argument('--extract_x_txt', type=nullORstr,
                                 default=None, help='Validation x text file')

        self.parser.add_argument('--extract_x_shorty', type=nullORstr,
                                 default=None, help='Validation x shorty file')

        self.parser.add_argument('--extract_x_img', type=nullORstr,
                                 default=None, help='Validation x image file')

        self.parser.add_argument('--extract_y', type=nullORstr, default=None,
                                 help='validation ground truth')

        self.parser.add_argument('--extract_fname', type=nullORstr,
                                 default="test.npy",
                                 help='Validation x ground truth')

        self.parser.add_argument('--filter_labels', type=nullORstr,
                                 help='Validation x ground truth')

        self.parser.add_argument('--preload', action='store_true',
                                 help='if preload')
        
        self.parser.add_argument('--save_all', action='store_true',
                                 help='save all matrix')

        self.parser.add_argument('--keep_all', action='store_true',
                                 help='keep all labels')

        self.parser.add_argument('--ranker', default="XAttnRanker",
                                 help='type of ranker to use')

        self.parser.add_argument('--encoder_init', default=None,
                                 help='init for ranker')
        
        self.parser.add_argument('--cosine_margin', default=0.5,
                                 help='cosine margin')
        
        self.parser.add_argument('--ignore_lbl_imgs', action='store_true',
                                 help='keep all labels')
        
        self.parser.add_argument('--validate', action='store_true',
                                 help='do validation')

        self.parser.add_argument('--module', type=int, action='store')
