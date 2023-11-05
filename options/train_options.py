from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--img_size', type=int, default=640, help='image size')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        
        # object detector training options
        parser.add_argument('--weights', type=str, default='', help='initial weights path')
        parser.add_argument('--cfg', type=str, default='yolov7.yaml', help='model.yaml path')
        parser.add_argument('--data-file', type=str, default='data/sentinel.yaml', help='data.yaml path')
        parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
        parser.add_argument('--epochs', type=int, default=300)
        parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
        parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
        parser.add_argument('--rect', action='store_true', help='rectangular training')
        parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
        parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
        parser.add_argument('--notest', action='store_true', help='only test final epoch')
        parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
        parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
        parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
        parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
        parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
        #parser.add_argument('--gpu-ids', default='0', help='cuda gpu IDs, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
        parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
        parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
        parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        parser.add_argument('--local-rank', type=int, default=-1, help='DDP parameter, do not modify')
        parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
        parser.add_argument('--project', default='runs/train', help='save to project/name')
        parser.add_argument('--entity', default=None, help='W&B entity')
        parser.add_argument('--name', default='exp', help='save to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--quad', action='store_true', help='quad dataloader')
        parser.add_argument('--linear-lr', action='store_true', help='linear LR')
        parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
        parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
        parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
        parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
        parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
        parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
        parser.add_argument('--polygon', default=False, action='store_true', help='enable polygon anchor boxes')
        parser.add_argument('--divide', default=False, action='store_true', help='divide images into patches')
        #parser.add_argument('--mode', type=str, help='dst source of tif files')
        parser.add_argument('--target', default='BRIDG', help='define the target objects')
        parser.add_argument('--warmup', default=100, help='how many epochs being trained only for object detection')
        parser.add_argument('--shp_path', default='./data/landmask/', help='shpfile for masking')
        parser.add_argument('--gan', default=False, action='store_true', help='include gan model')
        parser.add_argument('--subset', default='', help='data subset to train detector; None, A, or B')
        parser.add_argument('--fusion', default='early', help='enable data fusion i.e. early, mid, late')

        self.isTrain = False
        return parser
