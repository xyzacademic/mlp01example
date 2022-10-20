import argparse

parser = argparse.ArgumentParser(description='SCD01 Binary-classes')

# string
parser.add_argument('--target', default='', type=str,
                    help='checkpoint\'s name')
parser.add_argument('--type', default='scd', type=str,
                    help='scd or svm')
parser.add_argument('--source', default='svm.pkl', type=str,
                    help='checkpoint\'s name')
parser.add_argument('--gpu', default='1', type=str,
                    help='gpu device')
parser.add_argument('--metrics', default='balanced', type=str,
                    help='checkpoint\'s name')
parser.add_argument('--mc', default='mean', type=str,
                    help='checkpoint\'s name')
parser.add_argument('--ittype', default='one', type=str,
                    help='checkpoint\'s name')
parser.add_argument('--comment', default='nothing', type=str,
                    help='checkpoint\'s name')
parser.add_argument('--dataset', default='mnist', type=str,
                    help='dataset')
parser.add_argument('--version', default='v1', type=str,
                    help='scd version')
parser.add_argument('--init', default='normal', type=str,
                    help='initialization distribution')
parser.add_argument('--rdcnn-path', default='scd.pkl', type=str,
                    help='checkpoint\'s name')
parser.add_argument('--act', default='sign', type=str,
                    help='activation function')
parser.add_argument('--status', default='sigmoid', type=str,
                    help='activation function')
parser.add_argument('--loss', default='01', type=str,
                    help='random patch size')
parser.add_argument('--attack', default='fgsm', type=str,
                    help='random patch size')
parser.add_argument('--inc_version', default='v1', type=str,
                    help='random patch size')
parser.add_argument('--name', default='', type=str,
                    help='random patch size')
parser.add_argument('--adv_source', default=None, type=str,
                    help='random patch size')
parser.add_argument('--adv_structure', default='', type=str,
                    help='random patch size')
parser.add_argument('--pnorm', default='l2', type=str,
                    help='random patch size')
# int
parser.add_argument('--num-iters', default=100, type=int,
                    help='number of iters in each vote training')
parser.add_argument('--batch-size', default=256, type=int,
                    help='Batch size')
parser.add_argument('--updated-features', default=128, type=int,
                    help='number of features will be update in each iteration')
parser.add_argument('--round', default=1, type=int,
                    help='number of vote')
parser.add_argument('--interval', default=10, type=int,
                    help='number of neighbours will be considered '
                         'in bias choosing')
parser.add_argument('--n-jobs', default=5, type=int,
                    help='number of processes')
parser.add_argument('--num-gpus', default=1, type=int,
                    help='number of GPUs')
parser.add_argument('--hidden-nodes', default=5, type=int,
                    help='number of processes')
parser.add_argument('--iters', default=200, type=int,
                    help='ratio of rows in each iteration')
parser.add_argument('--epoch', default=20, type=int,
                    help='training epoch')
parser.add_argument('--aug-epoch', default=20, type=int,
                    help='attack epoch')
parser.add_argument('--train-size', default=200, type=int,
                    help='sample size')
parser.add_argument('--random-sign', default=0, type=int,
                    help='change lambda\'s sign')
parser.add_argument('--width', default=1, type=int,
                    help='number of iters in each vote training')
parser.add_argument('--updated-nodes', default=1, type=int,
                    help='change lambda\'s sign')
parser.add_argument('--h-times', default=1, type=int,
                    help='number of iters in each vote training')
parser.add_argument('--localit', default=100, type=int,
                    help='number of iters in each vote training')
parser.add_argument('--seed', default=2018, type=int,
                    help='random seed')
parser.add_argument('--oracle-size', default=256, type=int,
                    help='random seed')
parser.add_argument('--n_classes', default=2, type=int,
                    help='number of classes')
parser.add_argument('--patch-size', default=7, type=int,
                    help='random patch size')
parser.add_argument('--c0', default=7, type=int,
                    help='random patch size')
parser.add_argument('--c1', default=7, type=int,
                    help='random patch size')
parser.add_argument('--updated_conv_features', default=1, type=int,
                    help='random patch size')
parser.add_argument('--updated_fc_nodes', default=1, type=int,
                    help='random patch size')
parser.add_argument('--updated_conv_nodes', default=1, type=int,
                    help='random patch size')
parser.add_argument('--updated_fc_features', default=1, type=int,
                    help='random patch size')
parser.add_argument('--normalize', default=1, type=int,
                    help='random patch size')
parser.add_argument('--percentile', default=1, type=int,
                    help='random patch size')
parser.add_argument('--fail_count', default=1, type=int,
                    help='random patch size')
parser.add_argument('--index', default=1, type=int,
                    help='index of data points')
parser.add_argument('--votes', default=1, type=int,
                    help='number of votes')
parser.add_argument('--fc_diversity', default=0, type=int,
                    help='fc layer diversity training')
parser.add_argument('--conv_diversity', default=0, type=int,
                    help='conv layer diversity training')
parser.add_argument('--updated_conv_features_diversity', default=3, type=int,
                    help='diversity training, conv feature update')
parser.add_argument('--diversity_train_stop_iters', default=3000, type=int,
                    help='diversity training, conv feature update')
parser.add_argument('--diversity', default=0, type=int,
                    help='diversity training, conv feature update')
parser.add_argument('--no_bias', default=0, type=int,
                    help='diversity training, conv feature update')
parser.add_argument('--scale', default=1, type=int,
                    help='diversity training, conv feature update')
parser.add_argument('--target_class', default=1, type=int,
                    help='diversity training, conv feature update')
parser.add_argument('--updated_fc_ratio', default=1, type=int,
                    help='updated_fc_ratio')
parser.add_argument('--updated_conv_ratio', default=1, type=int,
                    help='diversity training, conv feature update')
parser.add_argument('--aug', default=0, type=int,
                    help='augmentation for data')
parser.add_argument('--adv_init', default=0, type=int,
                    help='augmentation for data')
parser.add_argument('--adv_train', default=0, type=int,
                    help='PGD adv train')
parser.add_argument('--cnn', default=0, type=int,
                    help='load data as image shape')
parser.add_argument('--mean_only', default=0, type=int,
                    help='load data as image shape')
parser.add_argument('--divmean', default=0, type=int,
                    help='divided 0.5 and minus 1')
parser.add_argument('--verbose_iter', default=20, type=int,
                    help='print iter')
parser.add_argument('--freeze_layer', default=0, type=int,
                    help='freeze training for layer')
parser.add_argument('--temp_save_per_iter', default=0, type=int,
                    help='save checkpoints every # iter')
parser.add_argument('--lr_decay_iter', default=0, type=int,
                    help='decay learning rate every # iter')
parser.add_argument('--batch_increase_iter', default=0, type=int,
                    help='batch_increase every # iter')
parser.add_argument('--balanced_sampling', default=1, type=int,
                    help='balanced sampling')
parser.add_argument('--bnn_layer', default=0, type=int,
                    help='bnn training layer')
parser.add_argument('--bp_layer', default=0, type=int,
                    help='bp training layer')
parser.add_argument('--reinit', default=0, type=int,
                    help='re-initialize layers weights and bias')
parser.add_argument('--bnn_layers', default=[], nargs='+', type=int,
                    help='bnn training layer')
parser.add_argument('--steps', default=10, type=int,
                    help='steps for adv train')
# float
parser.add_argument('--nrows', default=0.75, type=float,
                    help='ratio of rows in each iteration')
parser.add_argument('--alpha', default=0.00784, type=float,
                    help='ratio of rows in each iteration')
parser.add_argument('--nfeatures', default=1, type=float,
                    help='ratio of features in each vote')
parser.add_argument('--w-inc', default=0.17, type=float,
                    help='weights increments')
parser.add_argument('--w-inc1', default=0.17, type=float,
                    help='weights increments')
parser.add_argument('--w-inc2', default=0.02, type=float,
                    help='weights increments')
parser.add_argument('--eps', default=1, type=float,
                    help='epsilon in adversarial training')
parser.add_argument('--epsilon', default=1, type=float,
                    help='epsilon')
parser.add_argument('--Lambda', default=0.1, type=float,
                    help='ratio of features in each vote')
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate for substitute model')
parser.add_argument('--c', default=0.01, type=float,
                    help='c')
parser.add_argument('--b-ratio', default=0.2, type=float,
                    help='ratio of iterations of updating by balanced metrics')
parser.add_argument('--dropout', default=0.0, type=float,
                    help='ratio of dropout')

# bool
parser.add_argument('--adv-train', action='store_true',
                    help='Run adversarail training')
parser.add_argument('--no-eval', action='store_true',
                    help='evaluation')
parser.add_argument('--verbose', action='store_true',
                    help='show intermediate acc output')
parser.add_argument('--dual', action='store_true', help='Dual')
parser.add_argument('--save', action='store_true', help='Dual')
parser.add_argument('--resume', action='store_true', help='Resume')
parser.add_argument('--deep-search', action='store_true', help='Dual')
parser.add_argument('--alter-metrics', action='store_true', help='Dual')
parser.add_argument('--random-patch', action='store_true', help='Dual')
parser.add_argument('--normal-noise', action='store_true', help='Dual')
parser.add_argument('--binarize', action='store_true', help='Dual')
parser.add_argument('--fp16', action='store_true', help='float-precision 16')
parser.add_argument('--bootstrap', action='store_true', help='bootstrapping')
args = parser.parse_args()