# %%
import collections
import os
import sys
import time
from pprint import pformat

import ipdb
import numpy as np
import torch
import torch.autograd
import torch.nn as nn
from loguru import logger
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import InfiniteDataLoader
from dataset import ActivityDataset
from helpers import make_arg_parser
from net_utils import set_deterministic_and_get_rng
from nets import Classifier, Discriminator, SpatialTransformerBlock

logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> <level>{message}</level>")


class StranGAN(object):
    """
    STranGAN: Adversarially-learnt Spatial Transformer for scalable human activity recognition
    """

    def __init__(self, device, args):
        super(StranGAN, self).__init__()

        self.n_classes = args.n_classes
        self.n_channels = args.n_channels
        self.window_size = args.window_size
        self.device = device
        self.log_interval = args.log_interval

        self.classifier = Classifier(args.n_channels, args.n_classes).to(device)
        self.discriminator = Discriminator(args.n_channels).to(device)
        # self.generator = Generator(2, args.n_channels, args.window_size).to(device)
        # self.generator = nn.Sequential(
        #     *[SpatialTransformerBlock(args.n_channels, args.window_size) for i in range(2)]
        # ).to(device)
        self.generator = SpatialTransformerBlock(args.n_channels, args.window_size).to(device)

        logger.info(self.classifier)
        logger.info(self.discriminator)
        logger.info(self.generator)

        self.adversarial_loss = torch.nn.BCEWithLogitsLoss(reduction='mean').to(device)
        self.clf_loss = nn.NLLLoss().to(device)
        self.recon_loss = nn.SmoothL1Loss().to(device)
        # self.recon_loss = nn.MSELoss().to(device)

        self.optim_c = Adam(self.classifier.parameters(), lr=args.lr_FC,
                            betas=(args.lr_FC_b1, args.lr_FC_b2)
                            # amsgrad=True, weight_decay=1e-6
                            )

        """
            https://sthalles.github.io/advanced_gans/
            The discriminator trains with a learning rate 4 times greater than G - 0.004 and 0.001 respectively.
            A larger learning rate means that the discriminator will absorb a larger part of the gradient signal.
            Hence, a higher learning rate eases the problem of slow learning of the regularized discriminator.
            Also, this approach makes it possible to use the same rate of updates for the generator and
            the discriminator. In fact, we use a 1:1 update interval between generator and discriminator.
        """
        self.optim_d = SGD(self.discriminator.parameters(),
                           lr=args.lr_FD, weight_decay=1e-6)  # 0.000002, momentum=0.9
        # self.optim_d = Adam(self.discriminator.parameters(),
        #                    lr=args.lr_FD, weight_decay=1e-6)  # 0.000002, momentum=0.9
        self.optim_g = Adam(self.generator.parameters(),
                            lr=args.lr_G,
                            betas=(args.lr_G_b1, args.lr_G_b2),
                            amsgrad=False, weight_decay=1e-6)  # 0.0002

        # stochastic weight average
        # self.generator_swa = AveragedModel(self.generator)
        # self.scheduler = CosineAnnealingLR(self.optim_g, T_max=100)
        # self.swa_start = 5000
        # self.swa_scheduler = SWALR(self.optim_g, swa_lr=0.05)
        # self.scheduler_g = StepLR(self.optim_g, step_size=1000, gamma=0.5)

    @torch.no_grad()
    def test(self, model, test_loader, stage='train', generator=None):
        model.eval()
        if generator: generator.eval()
        loss = 0
        correct = 0
        y_true, y_pred = [], []

        for data, target in test_loader:
            data = data.to(self.device).float()
            target = target.to(self.device)
            y_true.append(target)

            if generator:
                data_, _ = generator(data)
                output = model(data_)
            else:
                output = model(data)

            # sum up batch loss
            loss += self.clf_loss(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            y_pred.append(pred.view_as(target))

            correct += pred.eq(target.view_as(pred)).sum().item()

        loss /= len(test_loader.dataset)
        acc = correct / len(test_loader.dataset)
        y_true = torch.cat(y_true, 0).cpu().numpy()
        y_pred = torch.cat(y_pred, 0).cpu().numpy()
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='micro')

        logger.info(
            f'CLF eval {stage} loss={loss:.6f} acc={acc * 100:3.2f}% {correct:5d}/{len(test_loader.dataset):5d} f1={f1:.4f}')

        return {
            'loss'     : loss,
            'acc'      : acc,
            'precision': precision,
            'recall'   : recall,
            'f1'       : f1,
            'support'  : support
        }

    def train_clf(self, source_loader_train, source_loader_val, target_loader_val, args):
        """
            Trains the source classifier
        """

        source_metrics_train = {}
        source_metrics_val = {}
        target_metrics_val = {}

        if args.clf_ckpt != '' and os.path.exists(args.clf_ckpt):
            logger.info(f'Loading Classifier from {args.clf_ckpt} ...')
            self.classifier.load_state_dict(torch.load(args.clf_ckpt))
            logger.success('Model loaded!')
            source_metrics_train = self.test(self.classifier, source_loader_train, 'source (train)')
            source_metrics_val = self.test(self.classifier, source_loader_val, 'source (val)  ')
            target_metrics_val = self.test(self.classifier, target_loader_val, 'target (val)  ')
        else:
            for epoch in range(1, args.n_epochs + 1):
                ts = time.time()
                self.classifier.train()
                for batch_idx, (X_source, y_source) in enumerate(source_loader_train):
                    X_source = X_source.to(self.device).float()
                    y_source = y_source.to(self.device)

                    self.optim_c.zero_grad()
                    y_source_pred = self.classifier(X_source)

                    loss_fc = self.clf_loss(y_source_pred, y_source)
                    loss_fc.backward()
                    self.optim_c.step()

                    if batch_idx % args.log_interval == 0:
                        logger.info(
                            f'CLF train epoch: {epoch:2d} {100. * batch_idx / len(source_loader_train):3.0f}%'
                            + f' {batch_idx * len(X_source):5d}/{len(source_loader_train.dataset)} lC={loss_fc.item():.6f}'
                        )
                te = time.time()
                logger.info(f'Took {(te - ts):.2f} seconds this epoch')
                logger.info('------------------------------------------------')
                source_metrics_train = self.test(self.classifier, source_loader_train, 'source (train)')
                source_metrics_val = self.test(self.classifier, source_loader_val, 'source (val)  ')
                target_metrics_val = self.test(self.classifier, target_loader_val, 'target (val)  ')
                logger.info('------------------------------------------------')

                save_path = os.path.join(args.save_dir, 'clf.pt')
                logger.info(f'Saving the Classifier in {save_path}')
                torch.save(self.classifier.state_dict(), save_path)

        return {
            'source-train': source_metrics_train,
            'source-val'  : source_metrics_val,
            'target-val'  : target_metrics_val
        }

    def train_gan(self, source_loader_da, target_loader_da, source_loader_clf_train,
                  source_loader_clf_val,
                  target_loader_clf_val, args):
        # ----------------------------
        #  First train the source classifier
        # ----------------------------
        self.train_clf(source_loader_clf_train, source_loader_clf_val, target_loader_clf_val, args)

        best_f1 = 0.0
        # check for resume
        if args.resume_gan:
            if args.gen_ckpt != '' and os.path.exists(args.gen_ckpt):
                logger.info(f'Loading Generator from {args.gen_ckpt} ...')
                self.generator.load_state_dict(torch.load(args.gen_ckpt))
                logger.success('Model loaded!')

            if args.dsc_ckpt != '' and os.path.exists(args.dsc_ckpt):
                logger.info(f'Loading Discriminator from {args.dsc_ckpt} ...')
                self.discriminator.load_state_dict(torch.load(args.dsc_ckpt))
                logger.success('Model loaded!')

            _ = self.test(self.classifier, target_loader_clf_val,
                          'target (xformed)', self.generator)
            best_f1 = _['f1']
            logger.info(f'Best result {best_f1}')

        # ----------------------------
        #  Now train the target network
        # ----------------------------
        valid = torch.ones(args.batch_size, 1, requires_grad=False).to(self.device) * args.soft_label_valid_disc
        fake = torch.ones(args.batch_size, 1, requires_grad=False).to(self.device) * args.soft_label_fake
        valid_alt = torch.ones(args.batch_size, 1, requires_grad=False).to(self.device) * args.soft_label_valid_gen

        source_iterator = iter(source_loader_da)
        target_iterator = iter(target_loader_da)

        step = 1
        while target_loader_da.epoch < args.gan_epochs:
            self.classifier.eval()
            self.discriminator.train()
            self.generator.train()

            X_source, y_source = next(source_iterator)
            X_target, y_target = next(target_iterator)

            X_source = X_source.to(self.device).float()
            y_source = y_source.to(self.device)
            X_target = X_target.to(self.device).float()
            y_target = y_target.to(self.device)

            # -----------------
            #  Train Generator
            # -----------------
            self.optim_g.zero_grad()
            X_gen, _ = self.generator(X_target)
            X_gen_source, _ = self.generator(X_source)

            loss_g_adv = self.adversarial_loss(self.discriminator(X_gen), valid_alt)
            loss_g_rec = self.recon_loss(X_gen_source, X_source)
            gamma = args.gamma
            # gamma = (np.e**((step-1)/1000)-1)/(np.e**((step-1)/1000)+1)
            # gamma = 0.95+ 0.05 * np.sin(step/100)
            # gamma = 0
            loss_g = loss_g_adv + loss_g_rec * gamma

            """
            workaround for the following error: 
                'RuntimeError: scatter_add_cuda_kernel does not have a deterministic implementation, 
                but you set 'torch.use_deterministic_algorithms(True)'. 
                You can turn off determinism just for this operation if that's acceptable for your application.'
            """
            torch.use_deterministic_algorithms(False)
            loss_g.backward()
            torch.use_deterministic_algorithms(True)

            self.optim_g.step()
            # self.scheduler_g.step()

            # SWA
            # if step % 1000 == 0:
            #     if step > self.swa_start:
            #         self.generator_swa.update_parameters(self.generator)
            #         self.swa_scheduler.step()
            #     else:
            #         self.scheduler.step()

            # --------------------------------
            #  Train the domain discriminator
            # --------------------------------
            self.optim_d.zero_grad()
            pred_valid = self.discriminator(X_source)
            pred_fake = self.discriminator(X_gen.detach())
            disc_acc = ((pred_valid.round().eq(valid.round()) * 1).sum().item() +
                        (pred_fake.round().eq(fake.round()) * 1).sum().item()) / (
                               args.batch_size * 2)
            loss_real = self.adversarial_loss(pred_valid, valid)
            loss_fake = self.adversarial_loss(pred_fake, fake)
            loss_d = (loss_real + loss_fake) / 2
            loss_d.backward()
            self.optim_d.step()

            if step % args.log_interval == 0:
                logger.info(
                    f"GAN tgt_epoch:{target_loader_da.epoch:3d} src_epoch:{source_loader_da.epoch:3d} step:{step:5d}"
                    + f" lD={loss_d.item():.4f}"
                    + f" lG={loss_g.item():.4f} lGr={loss_g_rec.item():.4f} lGa={loss_g_adv.item():.4f}"
                    + f" accD={disc_acc:.4f} gamma={gamma:.2f}")

            if step % args.eval_interval == 0:
                logger.info(
                    '------------------------------------------------------------------------------------------------------------')
                self.test(self.classifier, source_loader_clf_val, 'source (val+transformed)', self.generator)
                _ = self.test(self.classifier, target_loader_clf_val, 'target (val+transformed)', self.generator)
                logger.info(
                    '------------------------------------------------------------------------------------------------------------')

                if _['f1'] > best_f1:
                    logger.info('Updating best model!')
                    best_f1 = _['f1']
                    gen_save_path = os.path.join(args.save_dir, 'gen.pt')
                    dsc_save_path = os.path.join(args.save_dir, 'dsc.pt')
                    logger.info(f'Saving the generator and discriminator in folder {args.save_dir}')
                    torch.save(self.generator.state_dict(), gen_save_path)
                    torch.save(self.discriminator.state_dict(), dsc_save_path)
                    logger.success('Model saved!')
            step += 1

    @torch.no_grad()
    def interpret(self, source_loader, target_loader, args):
        """
        Save the transformed target samples and corresponding thetas for further analysis

        :param source_loader:
        :param target_loader:
        :param args:
        :return:
        """
        if args.clf_ckpt != '' and os.path.exists(args.clf_ckpt):
            logger.info(f'Loading Classifier from {args.clf_ckpt} ...')
            self.classifier.load_state_dict(torch.load(args.clf_ckpt))
            logger.success('Model loaded!')
        if args.gen_ckpt != '' and os.path.exists(args.gen_ckpt):
            logger.info(f'Loading Generator from {args.gen_ckpt} ...')
            self.generator.load_state_dict(torch.load(args.gen_ckpt))
            logger.success('Model loaded!')

        self.classifier.eval()
        self.generator.eval()

        thetas, target_data, xformed, source_data = [], [], [], []

        for data, target in target_loader:
            data = data.to(self.device).float()
            data_xformed, theta = self.generator(data)

            thetas.append(theta)
            target_data.append(data)
            xformed.append(data_xformed)

        for data, target in source_loader:
            data = data.to(self.device).float()
            source_data.append(data)

        thetas = torch.cat(thetas).cpu().numpy()
        target_data = torch.cat(target_data).cpu().numpy()
        source_data = torch.cat(source_data).cpu().numpy()
        xformed = torch.cat(xformed).cpu().numpy()

        theta_path = os.path.join(args.save_dir, 'thetas')
        logger.info('Saving theta, target, transformed target and source data to {}'.format(theta_path))
        np.savez_compressed(theta_path,
                            thetas=thetas, target_data=target_data, source_data=source_data, xformed=xformed)
        logger.success('Data saved!')


# %%

parser = make_arg_parser()
args = parser.parse_args()

rng, seed_worker = set_deterministic_and_get_rng(args)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

logger.add(os.path.join(args.save_dir, "training.log"))
logger.info(f'Current experiment parameters:\n{pformat(vars(args))}')

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with np.load(args.data_path, mmap_mode='r', allow_pickle=True) as npz:
    if args.subject_source.find(',') > 0:
        data_source = np.concatenate([
            npz['data_{}_{}'.format(ss, args.position_source)]
            for ss in tqdm(args.subject_source.split(','), 'creating source dataset')
        ])
    else:
        data_source = npz['data_{}_{}'.format(args.subject_source,
                                              args.position_source)]

    if args.subject_target.find(',') > 0:
        data_target = np.concatenate([
            npz['data_{}_{}'.format(st, args.position_target)]
            for st in tqdm(args.subject_target.split(','), 'creating target dataset')
        ])
    else:
        data_target = npz['data_{}_{}'.format(args.subject_target,
                                              args.position_target)]

source_train_dataset = ActivityDataset(data_source, args.window_size, args.n_channels, args.scaling,
                                       shuffle=False, train_set=True, train_frac=args.train_frac)
lencoder = source_train_dataset.lencoder
source_val_dataset = ActivityDataset(data_source, args.window_size, args.n_channels, args.scaling, lencoder=lencoder,
                                     shuffle=False, train_set=False, train_frac=args.train_frac)

target_train_dataset = ActivityDataset(data_target, args.window_size, args.n_channels, args.scaling,
                                       lencoder=lencoder, shuffle=False, train_set=True,
                                       train_frac=args.train_frac)
target_val_dataset = ActivityDataset(data_target, args.window_size, args.n_channels, args.scaling,
                                     lencoder=lencoder, shuffle=False, train_set=False,
                                     train_frac=args.train_frac)

# data loader for DA training
# -----------------------------------------------------------------------------------------------------------------------
source_loader_da = InfiniteDataLoader(source_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                      num_workers=args.num_workers, generator=rng, worker_init_fn=seed_worker)
target_loader_da = InfiniteDataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                      num_workers=args.num_workers, generator=rng, worker_init_fn=seed_worker)
# data loader for classification
# -----------------------------------------------------------------------------------------------------------------------
# training
source_loader_clf_train = DataLoader(source_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                     num_workers=args.num_workers, generator=rng, worker_init_fn=seed_worker)
# validation
source_loader_clf_val = DataLoader(source_val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                   num_workers=args.num_workers, generator=rng, worker_init_fn=seed_worker)
target_loader_clf_val = DataLoader(target_val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                   num_workers=args.num_workers, generator=rng, worker_init_fn=seed_worker)

strangan = StranGAN(device, args)
strangan.train_gan(source_loader_da, target_loader_da, source_loader_clf_train, source_loader_clf_val,
                   target_loader_clf_val, args)
strangan.interpret(source_loader_clf_val, target_loader_clf_val, args)
