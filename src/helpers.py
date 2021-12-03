import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_arg_parser():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    # dataset parameters
    # =========================================================================
    parser.add_argument('-d', '--data_path', required=True, help='path to dataset', type=str,
                        default='/workspace/phd/strangan/data/preprocessed/opportunity_all_users.npz')
    parser.add_argument('-ss', '--subject_source', required=True,
                        help='e.g. a-i for HHAR; provide multiple source with comma seperated values')
    parser.add_argument('-st', '--subject_target', required=True,
                        help='e.g. a-i for HHAR; provide multiple source with comma seperated values')
    parser.add_argument('-ps', '--position_source', help='None for HHAR')
    parser.add_argument('-pt', '--position_target', help='None for HHAR')
    parser.add_argument('-ch', '--n_channels', type=int, default=3,
                        help='Number of channels in the IMU steam')
    parser.add_argument('-cls', '--n_classes', type=int, default=6,
                        help='Number of classes in the activity data stream')
    parser.add_argument('-ws', '--window_size', type=int, default=128, help='Shape of the sliding window')
    parser.add_argument('--train_frac', type=float, default=0.7, help='Fraction of data to use for training')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for data loading')

    # training parameters
    # =========================================================================
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs for classifier')
    parser.add_argument('--gan_epochs', type=int, default=20,
                        help='Number of epochs (training dataset) in training GAN')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for random number generator both in numpy and pytorch')
    parser.add_argument('--log_interval', type=int, default=100, help='Interval for printing training info')
    parser.add_argument('--eval_interval', type=int, default=500, help='Interval for evaluating model')
    parser.add_argument('--gpu', default='0', help='GPU to use')

    # Hyperparameters
    # =========================================================================
    parser.add_argument('--scaling', type='bool', default=True, help='Whether to scale the data')
    parser.add_argument('--soft_label_valid_gen', type=float, default=0.9, help='Soft label for real data in generator')
    parser.add_argument('--soft_label_valid_disc', type=float, default=0.9,
                        help='Soft label for real data in discriminator')
    parser.add_argument('--soft_label_fake', type=float, default=0.0, help='Soft label for fake data in discriminator')

    # learning rates
    # adam for classifier
    # -------------------------------------------------------------------------
    parser.add_argument('--lr_FC', type=float, default=0.00002, help='Classifier learning rate')
    parser.add_argument('--lr_FC_b1', type=float, default=0.5, help='Classifier learning rate beta1')
    parser.add_argument('--lr_FC_b2', type=float, default=0.999, help='Classifier learning rate beta2')

    # SGD for discriminator
    # -------------------------------------------------------------------------
    parser.add_argument('--lr_FD', type=float, default=0.0002, help='Discriminator learning rate')

    # Adam for generator
    # -------------------------------------------------------------------------
    parser.add_argument('--lr_G', type=float, default=0.00002, help='Generator learning rate')
    parser.add_argument('--lr_G_b1', type=float, default=0.5, help='Generator learning rate beta1')
    parser.add_argument('--lr_G_b2', type=float, default=0.999, help='Generator learning rate beta2')

    # loss weights
    # -------------------------------------------------------------------------
    parser.add_argument('-wr', '--gamma', type=float, default=1.0, help='Source reconstruction weight')

    # output specifications
    # =========================================================================
    parser.add_argument('--save_dir', type=str, default='/tmp/strangan-runs/1', help='Directory to save results')
    parser.add_argument('--clf_ckpt', type=str, default='', help='absolute path to classifier checkpoint')
    parser.add_argument('--gen_ckpt', type=str, default='', help='absolute path to generator checkpoint')
    parser.add_argument('--dsc_ckpt', type=str, default='', help='absolute path to discriminator checkpoint')
    parser.add_argument('--resume_gan', type=bool, default=False, help='Whether to resume training GAN from checkpoint')
    return parser
