# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_

torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_start_method('spawn', force=True)

import os
import argparse
from itertools import chain
import numpy as np
from pathlib import Path
from tqdm import tqdm

from data import DatasetSet
from wavenet import WaveNet
from wavenet_models import cross_entropy_loss, Encoder, ZDiscriminator
from utils import create_output_dir, LossMeter, wrap
from midi_encoder import MidiEncoder, MultiHotMidiEncoder

import pdb

parser = argparse.ArgumentParser(description='PyTorch Code for A Universal Music Translation Network')
# Env options:
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 92)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--expName', type=str, required=True,
                    help='Experiment name')
parser.add_argument('--data',
                    metavar='D', type=Path, help='Data path', nargs='+')
parser.add_argument('--checkpoint', default='',
                    metavar='C', type=str, help='Checkpoint path')
parser.add_argument('--load-optimizer', action='store_true')
parser.add_argument('--per-epoch', action='store_true',
                    help='Save model per epoch')
parser.add_argument('--mode', type=int, default=3,
                    help='Mode of training to follow')
parser.add_argument('--multihot', type=int, default=1,
                    help='Use multihot encoding or not')
parser.add_argument('--pretraining_epochs', type=int, default=10,
                    help='number of epochs to freeze encoder decoder')

# Distributed
parser.add_argument('--dist-url', default='env://',
                    help='Distributed training parameters URL')
parser.add_argument('--dist-backend', default='nccl')
parser.add_argument('--local_rank', type=int,
                    help='Ignored during training.')

# Data options
parser.add_argument('--seq-len', type=int, default=16000,
                    help='Sequence length')
parser.add_argument('--epoch-len', type=int, default=10000,
                    help='Samples per epoch')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Batch size')
parser.add_argument('--num-workers', type=int, default=10,
                    help='DataLoader workers')
parser.add_argument('--data-aug', action='store_true',
                    help='Turns data aug on')
parser.add_argument('--magnitude', type=float, default=0.5,
                    help='Data augmentation magnitude.')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--lr-decay', type=float, default=0.98,
                    help='new LR = old LR * decay')
parser.add_argument('--short', action='store_true',
                    help='Run only a few batches per epoch for testing')
parser.add_argument('--h5-dataset-name', type=str, default='wav',
                    help='Dataset name in .h5 file')

# Encoder options
parser.add_argument('--latent-d', type=int, default=128,
                    help='Latent size')
parser.add_argument('--repeat-num', type=int, default=6,
                    help='No. of hidden layers')
parser.add_argument('--encoder-channels', type=int, default=128,
                    help='Hidden layer size')
parser.add_argument('--encoder-blocks', type=int, default=3,
                    help='No. of encoder blocks.')
parser.add_argument('--encoder-pool', type=int, default=800,
                    help='Number of encoder outputs to pool over.')
parser.add_argument('--encoder-final-kernel-size', type=int, default=1,
                    help='final conv kernel size')
parser.add_argument('--encoder-layers', type=int, default=10,
                    help='No. of layers in each encoder block.')
parser.add_argument('--encoder-func', type=str, default='relu',
                    help='Encoder activation func.')
parser.add_argument('--mode1-maxsize', type=int, default=16,
                    help='max number of chords to consider in a sequence in the given sampled interval for mode 1')


# Decoder options
parser.add_argument('--blocks', type=int, default=4,
                    help='No. of wavenet blocks.')
parser.add_argument('--layers', type=int, default=10,
                    help='No. of layers in each block.')
parser.add_argument('--kernel-size', type=int, default=2,
                    help='Size of kernel.')
parser.add_argument('--residual-channels', type=int, default=128,
                    help='Residual channels to use.')
parser.add_argument('--skip-channels', type=int, default=128,
                    help='Skip channels to use.')
parser.add_argument('--num-decoders', type=int, help='Number of decoders')

# Z discriminator options
parser.add_argument('--d-layers', type=int, default=3,
                    help='Number of 1d 1-kernel convolutions on the input Z vectors')
parser.add_argument('--d-channels', type=int, default=100,
                    help='1d convolutions channels')
parser.add_argument('--d-cond', type=int, default=1024,
                    help='WaveNet conditioning dimension')
parser.add_argument('--d-lambda', type=float, default=1e-2,
                    help='Adversarial loss weight.')
parser.add_argument('--p-dropout-discriminator', type=float, default=0.0,
                    help='Discriminator input dropout - if unspecified, no dropout applied')
parser.add_argument('--grad-clip', type=float,
                    help='If specified, clip gradients with specified magnitude')

# Midi Encoder options
parser.add_argument('--m-vocab-size', type=int, default=50000,
                    help='Total number of unique chords')
parser.add_argument('--m-hidden-size', type=int, default=960,
                    help='Hidden size of the LSTM')
parser.add_argument('--m-embed-size', type=int, default=20,
                    help='Embeddings size of the chords')
parser.add_argument('--m-lambda', type=float, default=5,
                    help='Aligned Loss Weight')

class Trainer:
    def __init__(self, args):
        self.args = args
        self.args.n_datasets = len(self.args.data)
        self.expPath = Path('checkpoints') / args.expName

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        self.logger = create_output_dir(args, self.expPath)
        self.data = [DatasetSet(d, args.seq_len, args) for d in args.data]

        # Get size of midi -> set the size of encoder accordingly
        for d in self.data:
            x, x_aug, x_midi = next(d.train_iter)
            if x_midi is not None:
                break
        self.midi_size = x_midi.size()


        assert not args.distributed or len(self.data) == int(
            os.environ['WORLD_SIZE']), "Number of datasets must match number of nodes"

        self.losses_recon = [LossMeter(f'recon {i}') for i in range(self.args.n_datasets)]
        self.loss_d_right = LossMeter('d')
        self.loss_total = LossMeter('total')
        self.loss_m_aligned = LossMeter('m')

        self.evals_recon = [LossMeter(f'recon {i}') for i in range(self.args.n_datasets)]
        self.eval_d_right = LossMeter('eval d')
        self.eval_total = LossMeter('eval total')

        self.encoder = Encoder(args)
        # self.decoder = WaveNet(args)
        if self.args.num_decoders:
            self.decoders = [WaveNet(args) for _ in range(args.num_decoders)]
        else:
            self.decoders = [WaveNet(args) for _ in self.data]
        self.discriminator = ZDiscriminator(args)
        # self.midi_encoder = MidiEncoder(args)
        
        # vocab size if basically number of notes
        # self.embeddings = nn.Embeddings(args.vocab_size, args.embedding_size)

        # if args.multihot == 1:
        #     self.midi_encoder = MultiHotMidiEncoder(args, self.midi_size[2])
        # else:
        #     self.midi_encoder = MidiEncoder(args)
        midi_encoder = MidiEncoder(args)

        self.start_epoch = 0
        
        #load pretrained model
        if args.checkpoint:
            print("Loading Pretrained models from ", args.checkpoint)
            checkpoint_args_path = os.path.dirname(args.checkpoint) + '/args.pth'
            checkpoint_args = torch.load(checkpoint_args_path)

            # self.start_epoch = checkpoint_args[-1] + 1
            
            states = torch.load(args.checkpoint)
            print(states.keys())
            #load encoder
            self.encoder.load_state_dict(states['encoder_state'])
            print("Encoder loaded")
            #load discriminator
            if 'discriminator_state' in states:
                self.discriminator.load_state_dict(states['discriminator_state'])
                print("Discriminator loaded")
            #load midi encoder
            if 'midi_encoder_state' in states:
                midi_encoder_state = states['midi_encoder_state']
                module_keys = []
                for k in midi_encoder_state.keys():
                    if k[:7] == 'module.':
                        module_keys.append(k)
                for k in module_keys:
                    midi_encoder_state[k[7:]] = midi_encoder_state[k]
                    del midi_encoder_state[k]
                self.midi_encoder.load_state_dict(states['midi_encoder_state'])
                print("Midi Encoder loaded")
            #load decoders
            for i, decoder in enumerate(self.decoders):
                parent = os.path.dirname(args.checkpoint)
                if i>=args.num_decoders-1:
                    self.decoders[i].load_state_dict(torch.load(parent + f'/d_1.pth')['decoder_state']) #pick piano decoder
                else:
                    self.decoders[i].load_state_dict(torch.load(parent + f'/d_{i}.pth')['decoder_state'])
            print("Decoders loaded")

            self.logger.info('Loaded checkpoint parameters')


        if args.distributed:
            work = 0
            wont = 0
            this = wont+work
            # self.encoder.cuda()
            # self.encoder = torch.nn.parallel.DistributedDataParallel(self.encoder)
            # self.discriminator.cuda()
            # self.discriminator = torch.nn.parallel.DistributedDataParallel(self.discriminator)
            # self.logger.info('Created DistributedDataParallel')
            # self.decoder = torch.nn.DataParallel(self.decoder).cuda()
        else:
            self.encoder = torch.nn.DataParallel(self.encoder).cuda()
            self.discriminator = torch.nn.DataParallel(self.discriminator).cuda()
            self.midi_encoder = torch.nn.DataParallel(self.midi_encoder).cuda()
            # if self.onehot:
            #     self.onehot_midi_encoder = torch.nn.DataParallel(self.onehot_midi_encoder).cuda()
        self.decoders = [torch.nn.DataParallel(d).cuda() for d in self.decoders]

        # self.model_optimizer = optim.Adam(chain(self.encoder.parameters(),
        #                                         self.decoder.parameters(),
        #                                         self.midi_encoder.parameters()),
        #                                   lr=args.lr)

        params = [{'params' : d.parameters()} for d in self.decoders] + [{'params': self.encoder.parameters()}] +\
        [{'params': self.midi_encoder.parameters()}]
        
        self.model_optimizer = optim.Adam(params, lr=args.lr)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(),
                                      lr=args.lr)

        if args.checkpoint and args.load_optimizer:
            self.model_optimizer.load_state_dict(states['model_optimizer_state'])
            self.d_optimizer.load_state_dict(states['d_optimizer_state'])

        self.lr_manager = torch.optim.lr_scheduler.ExponentialLR(self.model_optimizer, args.lr_decay)
        self.lr_manager.last_epoch = self.start_epoch
        self.lr_manager.step()

    def train_batch(self, epoch, x, x_aug, x_midi=None, dset_num=None):
        # print(x)
        x, x_aug= x.float(), x_aug.float()
        assert(dset_num is not None)
        
        if epoch < self.args.pretraining_epochs:
            for p in self.encoder.parameters():
                p.requires_grad=False
            for decoder in self.decoders:
                for p in decoder.parameters():
                    p.requires_grad=False
        
        # Optimize D - discriminator right
        z = self.encoder(x)
        z_logits = self.discriminator(z)
        discriminator_right = F.cross_entropy(z_logits, torch.tensor([dset_num] * x.size(0)).long().cuda()).mean()
        loss = discriminator_right * self.args.d_lambda

        self.loss_d_right.add(discriminator_right.data.item())
        self.d_optimizer.zero_grad()
        loss.backward()
        if self.args.grad_clip is not None:
            clip_grad_value_(self.discriminator.parameters(), self.args.grad_clip)

        self.d_optimizer.step()

        # optimize G - reconstructs well, discriminator wrong
        z = self.encoder(x_aug)
        if dset_num < self.args.num_decoders-1:
            y = self.decoders[dset_num](x, z)
        else:
            y = self.decoders[-1](x, z)
        z_logits = self.discriminator(z)
        discriminator_wrong = - F.cross_entropy(z_logits, torch.tensor([dset_num] * x.size(0)).long().cuda()).mean()

        if not (-100 < discriminator_right.data.item() < 100):
            self.logger.debug(f'z_logits: {z_logits.detach().cpu().numpy()}')
            self.logger.debug(f'dset_num: {dset_num}')

        recon_loss = cross_entropy_loss(y, x)
        self.losses_recon[dset_num].add(recon_loss.data.cpu().numpy().mean())

        loss = (recon_loss.mean() + self.args.d_lambda * discriminator_wrong)
        # print("recon loss:", recon_loss.mean().item()," discriminator loss:",discriminator_wrong.item())
        
        aligned_loss = 0.0
        if x_midi is not None:
            h, _  = self.midi_encoder(x_midi) # size : (bs, hidden_size)
            h = h.view(z.shape)
            aligned_loss = F.mse_loss(h, z)
            loss += self.args.m_lambda * aligned_loss
            self.loss_m_aligned.add(aligned_loss.data.item())

        self.model_optimizer.zero_grad()
        loss.backward()
        if self.args.grad_clip is not None:
            clip_grad_value_(self.encoder.parameters(), self.args.grad_clip)
            for decoder in self.decoders:
                clip_grad_value_(decoder.parameters(), self.args.grad_clip)
        self.model_optimizer.step()

        self.loss_total.add(loss.data.item())

        if epoch < self.args.pretraining_epochs:
            for p in self.encoder.parameters():
                p.requires_grad=True
            for decoder in self.decoders:
                for p in decoder.parameters():
                    p.requires_grad=True

        return loss.data.item()

    def eval_batch(self, x, x_aug, x_midi, dset_num):
        x, x_aug = x.float(), x_aug.float()
        
        assert(dset_num is not None)

        z = self.encoder(x)
        if dset_num < self.args.num_decoders-1:
            y = self.decoders[dset_num](x, z)
        else:
            y = self.decoders[-1](x, z)
        z_logits = self.discriminator(z)

        z_classification = torch.max(z_logits, dim=1)[1]

        z_accuracy = (z_classification == dset_num).float().mean()

        self.eval_d_right.add(z_accuracy.data.item())

        # discriminator_right = F.cross_entropy(z_logits, dset_num).mean()
        discriminator_right = F.cross_entropy(z_logits, torch.tensor([dset_num] * x.size(0)).long().cuda()).mean()
        recon_loss = cross_entropy_loss(y, x)

        self.evals_recon[dset_num].add(recon_loss.data.cpu().numpy().mean())
        
        total_loss = discriminator_right.data.item() * self.args.d_lambda + \
                     recon_loss.mean().data.item()
                     
        aligned_loss = 0.0
        if x_midi is not None:
            h, _  = self.midi_encoder(x_midi) # size : (bs, hidden_size)
            h = h.view(z.shape)
            aligned_loss = F.mse_loss(h, z)
            total_loss += self.args.m_lambda * aligned_loss.mean().data.item()


        self.eval_total.add(total_loss)

        return total_loss

    def train_epoch(self, epoch):
        for meter in self.losses_recon:
            meter.reset()
        self.loss_d_right.reset()
        self.loss_total.reset()
        self.loss_m_aligned.reset()

        self.encoder.train()
        for decoder in self.decoders:
            decoder.train()
        self.discriminator.train()
        self.midi_encoder.train()
        n_batches = self.args.epoch_len

        with tqdm(total=n_batches, desc='Train epoch %d' % epoch) as train_enum:
            for batch_num in range(n_batches):
                if self.args.short and batch_num == 3:
                    break

                if self.args.distributed:
                    assert self.args.rank < self.args.n_datasets, "No. of workers must be equal to #dataset"
                    # dset_num = (batch_num + self.args.rank) % self.args.n_datasets
                    dset_num = self.args.rank
                else:
                    dset_num = batch_num % self.args.n_datasets

                # pdb.set_trace()
                x, x_aug, x_midi = next(self.data[dset_num].train_iter)
                # print(next(self.data[dset_num].train_iter))
                # print("x: ", x)
                # print("x_midi: ",x_midi.size())

                x = wrap(x)
                x_aug = wrap(x_aug)
                if(x_midi is not None):
                    x_midi = wrap(x_midi)
                batch_loss = self.train_batch(epoch, x, x_aug, x_midi, dset_num)

                train_enum.set_description(f'Train (loss: {batch_loss:.2f}) epoch {epoch}')
                train_enum.update()

    def evaluate_epoch(self, epoch):
        for meter in self.evals_recon:
            meter.reset()
        self.eval_d_right.reset()
        self.eval_total.reset()

        self.encoder.eval()
        for decoder in self.decoders:
            decoder.eval()
        self.discriminator.eval()
        # if self.onehot:
        #     self.onehot_midi_encoder.eval()
        # else:
        self.midi_encoder.eval()

        n_batches = int(np.ceil(self.args.epoch_len / 10))

        with tqdm(total=n_batches) as valid_enum, \
                torch.no_grad():
            for batch_num in range(n_batches):
                if self.args.short and batch_num == 10:
                    break

                if self.args.distributed:
                    assert self.args.rank < self.args.n_datasets, "No. of workers must be equal to #dataset"
                    dset_num = self.args.rank
                else:
                    dset_num = batch_num % self.args.n_datasets

                x, x_aug, x_midi = next(self.data[dset_num].train_iter)

                x = wrap(x)
                x_aug = wrap(x_aug)
                if(x_midi is not None):
                    x_midi = wrap(x_midi)
                
                batch_loss = self.eval_batch(x, x_aug, x_midi, dset_num)

                valid_enum.set_description(f'Test (loss: {batch_loss:.2f}) epoch {epoch}')
                valid_enum.update()

    @staticmethod
    def format_losses(meters):
        losses = [meter.summarize_epoch() for meter in meters]
        return ', '.join('{:.4f}'.format(x) for x in losses)

    def train_losses(self):
        meters = [*self.losses_recon, self.loss_d_right, self.loss_m_aligned]
        return self.format_losses(meters)

    def eval_losses(self):
        meters = [*self.evals_recon, self.eval_d_right]
        return self.format_losses(meters)

    def train(self):
        best_eval = float('inf')

        # Begin!
        for epoch in range(self.start_epoch, self.start_epoch + self.args.epochs):
            self.logger.info(f'Starting epoch, Rank {self.args.rank}, Dataset: {self.args.data[self.args.rank]}')
            self.train_epoch(epoch)
            self.evaluate_epoch(epoch)

            self.logger.info(f'Epoch %s Rank {self.args.rank} - Train loss: (%s), Test loss (%s)',
                             epoch, self.train_losses(), self.eval_losses())
            self.lr_manager.step()
            val_loss = self.eval_total.summarize_epoch()

            if val_loss < best_eval:
                self.save_model(f'bestmodel_{self.args.rank}.pth')
                best_eval = val_loss
                for i, decoder in enumerate(self.decoders):
                    decoder_path = self.expPath/f'd_{i}.pth'
                    torch.save({'decoder_state': decoder.module.state_dict()},
                            decoder_path)

            if not self.args.per_epoch:
                self.save_model(f'lastmodel_{self.args.rank}.pth')
            else:
                self.save_model(f'lastmodel_{epoch}_rank_{self.args.rank}.pth')

            if self.args.is_master:
                torch.save([self.args,
                            epoch],
                           '%s/args.pth' % self.expPath)

            self.logger.debug('Ended epoch')

    def save_model(self, filename):
        save_path = self.expPath / filename

        torch.save({'encoder_state': self.encoder.module.state_dict(),
                    'discriminator_state': self.discriminator.module.state_dict(),
                    'model_optimizer_state': self.model_optimizer.state_dict(),
                    'dataset': self.args.rank,
                    'd_optimizer_state': self.d_optimizer.state_dict(),
                    'midi_encoder_state': self.midi_encoder.state_dict()
                    },
                   save_path)
        
        # for i, decoder in enumerate(self.decoders):
        #     # decoder_path = str(self.expPath) + "_d_" + str(i) + ".pth"
        #     decoder_filename = "decoder_" + str(i) + ".pth"
        #     decoder_path = self.expPath / decoder_filename
        #     torch.save({'decoder_state': decoder.module.state_dict()},
        #                decoder_path)

        self.logger.debug(f'Saved model to {save_path}')


def main():
    args = parser.parse_args()
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        # import ipdb; ipdb.set_trace()
        # if 'MASTER_ADDR' not in os.environ:
        #     var = os.environ["SLURM_NODELIST"]
        #     match = re.match(r'learnfair\[(\d+).*', var)
        #     master_id = match.group(1)
        #     os.environ["MASTER_ADDR"] = "learnfair" + master_id
        #     print('Set MASTER_ADDR to', os.environ['MASTER_ADDR'])
        if int(os.environ['RANK']) == 0:
            args.is_master = True
        else:
            args.is_master = False
        args.rank = int(os.environ['RANK'])

        print('Before init_process_group')
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url)
    else:
        args.rank = 0
        args.is_master = True

    Trainer(args).train()


if __name__ == '__main__':
    main()
