# Created on 2019/01
# Author: Kaituo XU (NPU-ASLP)

import os
import time

import torch


class Solver(object):
    
    def __init__(self, data_loader, model, criterion, optimizer, args):
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        # Training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.max_norm = args.max_norm
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        # self.model_path = args.model_path
        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self.visdom = args.visdom
        self.visdom_epoch = args.visdom_epoch
        self.visdom_id = args.visdom_id
        if self.visdom:
            from visdom import Visdom
            self.vis = Visdom(env=self.visdom_id)
            self.vis_opts = dict(title=self.visdom_id,
                                 ylabel='Loss', xlabel='Epoch',
                                 legend=['train loss', 'cv loss'])
            self.vis_window = None
            self.vis_epochs = torch.arange(1, self.epochs + 1)

        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from)
            self.model.module.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()

            tr_avg_loss = self._run_one_epoch(epoch)

            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            # Save model each epoch
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = tr_avg_loss  # Fake
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                if self.use_cuda:
                    torch.save(self.model.module.serialize(self.model.module,
                                                        self.optimizer, epoch + 1,
                                                        tr_loss=self.tr_loss,
                                                        cv_loss=self.cv_loss),
                            file_path)
                else:
                    torch.save(self.model.serialize(self.model,
                                                    self.optimizer, epoch + 1,
                                                    tr_loss=self.tr_loss,
                                                    cv_loss=self.cv_loss),
                            file_path)
                print('Saving checkpoint model to %s' % file_path)

            # visualizing loss using visdom
            if self.visdom:
                x_axis = self.vis_epochs[0:epoch + 1]
                y_axis = torch.stack(
                    (self.tr_loss[0:epoch + 1], self.cv_loss[0:epoch + 1]), dim=1)
                if self.vis_window is None:
                    self.vis_window = self.vis.line(
                        X=x_axis,
                        Y=y_axis,
                        opts=self.vis_opts,
                    )
                else:
                    self.vis.line(
                        X=x_axis.unsqueeze(0).expand(y_axis.size(
                            1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
                        Y=y_axis,
                        win=self.vis_window,
                        update='replace',
                    )

    def _run_one_epoch(self, epoch):
        start = time.time()
        total_loss = 0

        data_loader = self.data_loader

        # visualizing loss using visdom
        if self.visdom_epoch:
            vis_opts_epoch = dict(title=self.visdom_id + " epoch " + str(epoch),
                                  ylabel='Loss', xlabel='Epoch')
            vis_window_epoch = None
            vis_iters = torch.arange(1, len(data_loader) + 1)
            vis_iters_loss = torch.Tensor(len(data_loader))

        for i, (data) in enumerate(data_loader):
            text_padded, input_lengths, feat_padded, stop_token_padded, output_lengths = data
            if self.use_cuda:
                text_padded = text_padded.cuda()
                input_lengths = input_lengths.cuda()
                feat_padded = feat_padded.cuda()
                stop_token_padded = stop_token_padded.cuda()
                output_lengths = output_lengths.cuda()
            y_pred = self.model(text_padded, input_lengths, feat_padded, output_lengths)
            y_target = (feat_padded, stop_token_padded)
            loss = self.criterion(y_pred, y_target)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optimizer.step()

            total_loss += loss.item()

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                      flush=True)

            # visualizing loss using visdom
            if self.visdom_epoch:
                vis_iters_loss[i] = loss.item()
                if i % self.print_freq == 0:
                    x_axis = vis_iters[:i+1]
                    y_axis = vis_iters_loss[:i+1]
                    if vis_window_epoch is None:
                        vis_window_epoch = self.vis.line(X=x_axis, Y=y_axis,
                                                         opts=vis_opts_epoch)
                    else:
                        self.vis.line(X=x_axis, Y=y_axis, win=vis_window_epoch,
                                      update='replace')

        return total_loss / (i + 1)
