"""
Wrapper functions for the networks
"""
# Built-in
import os
import time

# Torch
import torch
from torch import nn
import torch.nn.functional as F
from torch import pow, add, mul, div, sqrt, abs, square, conj
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
from torchsummary import summary
from torch.optim import lr_scheduler
from torchviz import make_dot
from utils.plotting import plot_weights_3D, plotMSELossDistrib, \
    compare_spectra, compare_spectra_with_params, plot_complex, plot_all
from utils.logging import evaluate_mse_on_dataset

# Libs
import matplotlib
matplotlib.use('Agg')
import numpy as np

import warnings
warnings.filterwarnings('ignore')

class Network(object):
    def __init__(self, model_fn, flags, train_loader, test_loader,
                 ckpt_dir=os.path.join(os.path.abspath(''), 'models'),
                 inference_mode=False, saved_model=None):
        self.model_fn = model_fn                                # The network architecture object
        self.flags = flags                                      # The flags containing the hyperparameters
        if inference_mode:                                      # If inference mode, use saved model
            self.ckpt_dir = os.path.join(ckpt_dir, saved_model)
            self.saved_model = saved_model
            print("This is inference mode, the ckpt is", self.ckpt_dir)
        else:                                                   # Network training mode, create a new ckpt folder
            if flags.model_name is None:                    # Use custom name if possible, otherwise timestamp
                self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
            else:
                self.ckpt_dir = os.path.join(ckpt_dir, flags.model_name)
        self.model = self.create_model()                        # The model itself
        self.init_weights()
        self.loss = self.make_custom_loss()                            # The loss function
        self.optm = None                                        # The optimizer: Initialized at train()
        self.lr_scheduler = None                                # The lr scheduler: Initialized at train()
        self.train_loader = train_loader                        # The train data loader
        self.test_loader = test_loader                          # The test data loader
        self.log = SummaryWriter(self.ckpt_dir)     # Create a summary writer for tensorboard
        self.best_validation_loss = float('inf')    # Set the BVL to large number
        self.best_mse_loss = float('inf')           # Set the mse loss to large number

    def create_model(self):
        """
        Function to create the network module from provided model fn and flags
        :return: the created nn module
        """
        model = self.model_fn(self.flags)
        print(model)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('There are %d trainable out of %d total parameters' %(pytorch_total_params, pytorch_total_params_train))
        return model

    def make_MSE_loss(self, logit=None, labels=None):
        """
        Create a tensor that represents the loss. This is consistent both at training time \
        and inference time for a backward model
        :param logit: The output of the network
        :return: the total loss
        """
        if logit is None:
            return None
        MSE_loss = nn.functional.mse_loss(logit, labels, reduction='mean')          # The MSE Loss of the network
        MSE_loss *= 10000                   # Hyperparameter for tuning the strength of the mse loss

        return MSE_loss

    def make_custom_loss(self, logit1=None, logit2=None, labels=None):
        """
        Create a custom loss function as an alternative to the conventional MSE loss.
        :param logit: The output of the network
        :return: the total loss
        """
        if logit1 is None:
            return None
        # loss1 = nn.functional.mse_loss(logit1.real.float(), labels[:, : ,0].real.float(), reduction='mean')
        # loss2 = nn.functional.mse_loss(logit1.imag.float(), labels[:, :, 0].imag.float(), reduction='mean')
        # loss3 = nn.functional.mse_loss(logit2.real.float(), labels[:, :, 1].real.float(), reduction='mean')
        # loss4 = nn.functional.mse_loss(logit2.imag.float(), labels[:, :, 1].imag.float(), reduction='mean')
        # custom_loss = loss1 + loss2 + loss3 + loss4
        # custom_loss *= self.flags.loss_factor

        loss1 = nn.functional.mse_loss(square(abs(logit1)).float(), square(abs(labels[:, :, 0])).float(), reduction='mean')
        loss2 = nn.functional.mse_loss(square(abs(logit2)).float(), square(abs(labels[:, :, 1])).float(), reduction='mean')
        loss3 = nn.functional.mse_loss(logit1.imag.float(), labels[:, :, 0].imag.float(), reduction='mean')
        # loss4 = nn.functional.mse_loss(logit2.imag.float(), labels[:, :, 1].imag.float(), reduction='mean')
        custom_loss = loss1 + loss2 + loss3
        custom_loss *= self.flags.loss_factor
        return custom_loss

    def init_weights(self):
        """
        Initialize Lorentz layer weights to start within a certain range. Significantly affects training performance
        due to exploding/vanishing gradients, especially early in training.
        :return: None
        """
        for layer_name, child in self.model.named_children():
            for param in self.model.parameters():
                if ('_w0' in layer_name):
                    torch.nn.init.uniform_(child.weight, a=0.0, b=1.5)
                    # torch.nn.init.xavier_uniform_(child.weight)
                elif ('_wp' in layer_name):
                    torch.nn.init.uniform_(child.weight, a=0.0, b=0.3)
                    # torch.nn.init.xavier_uniform_(child.weight)
                elif ('_g' in layer_name):
                    torch.nn.init.uniform_(child.weight, a=0.0, b=0.1)
                    # torch.nn.init.xavier_uniform_(child.weight)
                elif ('_inf' in layer_name):
                    torch.nn.init.uniform_(child.weight, a=0.0, b=0.03)
                    # torch.nn.init.xavier_uniform_(child.weight)
                else:
                    if ((type(child) == nn.Linear) | (type(child) == nn.Conv2d)):
                        torch.nn.init.xavier_uniform_(child.weight)
                        if child.bias:
                            child.bias.data.fill_(0.00)

    def add_network_noise(self):
        """
        Adds noise to Lorentz layer network weights with amplitude specified as a hyperparameter
        :return: None
        """
        with torch.no_grad():
            # for param in self.model.parameters():
            #     param.add_(torch.randn(param.size()).cuda() * self.flags.ntwk_noise)

            for layer_name, child in self.model.named_children():
                for param in self.model.parameters():
                    if ('_w0' in layer_name):
                        param.add_(torch.randn(param.size()).cuda() * self.flags.ntwk_noise)
                    elif ('_wp' in layer_name):
                        param.add_(torch.randn(param.size()).cuda() * self.flags.ntwk_noise)
                    elif ('_g' in layer_name):
                        param.add_(torch.randn(param.size()).cuda() * self.flags.ntwk_noise)
                    elif ('_inf' in layer_name):
                        pass
                    else:
                        # if ((type(child) == nn.Linear) | (type(child) == nn.Conv2d)):
                        #     param.add_(torch.randn(param.size()).cuda() * self.flags.ntwk_noise)
                        pass

    def make_optimizer(self):
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed.
        :return:
        """
        if self.flags.optim == 'Adam':
            op = torch.optim.Adam(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'AdamW':
            op = torch.optim.AdamW(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'Adamax':
            op = torch.optim.Adamax(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SparseAdam':
            op = torch.optim.SparseAdam(self.model.parameters(), lr=self.flags.lr)
        elif self.flags.optim == 'RMSprop':
            op = torch.optim.RMSprop(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SGD':
            op = torch.optim.SGD(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale, momentum=0.9, nesterov=True)
        elif self.flags.optim == 'LBFGS':
            op = torch.optim.LBFGS(self.model.parameters(), lr=1, max_iter=20, history_size=100)

        else:
            raise Exception("Optimizer is not available at the moment.")
        return op

    def make_lr_scheduler(self):
        """
        Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
        1. StepLR (decrease lr in stepwise fashion)
        2. ReduceLROnPlateau (decrease lr when validation error stops improving)
        :return:
        """
        # return lr_scheduler.StepLR(optimizer=self.optm, step_size=50, gamma=0.75, last_epoch=-1)
        return lr_scheduler.ReduceLROnPlateau(optimizer=self.optm, mode='min',
                                        factor=self.flags.lr_decay_rate,
                                          patience=10, verbose=True, threshold=1e-4)

    def save(self):
        """
        Saving the model to the current check point folder with name best_model.pt
        :return: None
        """
        torch.save(self.model, os.path.join(self.ckpt_dir, 'best_model.pt'))

    def load(self):
        """
        Loading the model from the check point folder with name best_model.pt
        :return:
        """
        self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model.pt'))

    def evaluate(self, save_dir='eval/'):
        """
        Evaluates model and calculates MSE on test dataset. Currently set to output R,T rather than r,t
        :return: truth and prediction files from model
        """

        self.load()                             # load the model as constructed
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        self.model.eval()                       # Evaluation mode

        # Get the file names
        Ypred_T_file = os.path.join(save_dir, 'test_Ypred_T_{}.csv'.format(self.saved_model))
        Ypred_R_file = os.path.join(save_dir, 'test_Ypred_R_{}.csv'.format(self.saved_model))
        Xtruth_file = os.path.join(save_dir, 'test_Xtruth_{}.csv'.format(self.saved_model))
        Ytruth_T_file = os.path.join(save_dir, 'test_Ytruth_T_{}.csv'.format(self.saved_model))
        Ytruth_R_file = os.path.join(save_dir, 'test_Ytruth_R_{}.csv'.format(self.saved_model))

        # Open those files to append
        with open(Xtruth_file, 'a') as fxt,open(Ytruth_T_file, 'a') as fyt_1, open(Ypred_T_file, 'a') as fyp_1, \
                open(Ytruth_R_file, 'a') as fyt_2, open(Ypred_R_file, 'a') as fyp_2:
            # Loop through the eval data and evaluate
            with torch.no_grad():
                for ind, (geometry, spectra) in enumerate(self.test_loader):
                    if cuda:
                        geometry = geometry.cuda()
                        spectra = spectra.cuda()
                    pred_r, pred_t = self.model(geometry)  # Get the output

                    T_pred = square(abs(pred_t)).cpu().data.numpy()
                    R_pred = square(abs(pred_r)).cpu().data.numpy()
                    T_truth = square(abs(spectra[:, :, 1])).cpu().data.numpy()
                    R_truth = square(abs(spectra[:, :, 0])).cpu().data.numpy()

                    np.savetxt(fxt, geometry.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(fyt_1, T_truth, fmt='%.3f')
                    np.savetxt(fyp_1, T_pred, fmt='%.3f')
                    np.savetxt(fyt_2, R_truth, fmt='%.3f')
                    np.savetxt(fyp_2, R_pred, fmt='%.3f')

        mse = evaluate_mse_on_dataset(Ytruth_T_file, Ypred_T_file)
        print('MSE is %.7f' % mse)

        return Ypred_T_file, Ytruth_T_file, Ypred_R_file, Ytruth_R_file,

    def evaluate_lorentz_params(self, save_dir='eval/'):
        """
        Evaluates model and calculates the Lorentz parameters for a test dataset.
        :return: Epsilon and Mu parameters from model
        """

        self.load()                             # load the model as constructed
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        self.model.eval()                       # Evaluation mode

        # Get the file names
        eps_wp_file = os.path.join(save_dir, 'test_eps_wp_{}.csv'.format(self.saved_model))
        eps_w0_file = os.path.join(save_dir, 'test_eps_w0_{}.csv'.format(self.saved_model))
        eps_g_file = os.path.join(save_dir, 'test_eps_g_{}.csv'.format(self.saved_model))
        eps_inf_file = os.path.join(save_dir, 'test_eps_inf_{}.csv'.format(self.saved_model))
        mu_wp_file = os.path.join(save_dir, 'test_mu_wp_{}.csv'.format(self.saved_model))
        mu_w0_file = os.path.join(save_dir, 'test_mu_w0_{}.csv'.format(self.saved_model))
        mu_g_file = os.path.join(save_dir, 'test_mu_g_{}.csv'.format(self.saved_model))
        mu_inf_file = os.path.join(save_dir, 'test_mu_inf_{}.csv'.format(self.saved_model))

        # Open those files to append
        with open(eps_wp_file, 'a') as ewpf,open(eps_w0_file, 'a') as ew0f, open(eps_g_file, 'a') as egf, \
                open(eps_inf_file, 'a') as eif, open(mu_wp_file, 'a') as mwpf,open(mu_w0_file, 'a') as mw0f, \
                open(mu_g_file, 'a') as mgf, open(mu_inf_file, 'a') as mif:
            # Loop through the eval data and evaluate
            with torch.no_grad():
                for ind, (geometry, spectra) in enumerate(self.test_loader):
                    if cuda:
                        geometry = geometry.cuda()
                        spectra = spectra.cuda()
                    pred_r, pred_t = self.model(geometry)  # Get the output

                    eps_wps = self.model.eps_params_out[1].cpu().data.numpy()
                    eps_w0s = self.model.eps_params_out[0].cpu().data.numpy()
                    eps_gs = self.model.eps_params_out[2].cpu().data.numpy()
                    eps_infs = self.model.eps_params_out[3].cpu().data.numpy()

                    mu_wps = self.model.mu_params_out[1].cpu().data.numpy()
                    mu_w0s = self.model.mu_params_out[0].cpu().data.numpy()
                    mu_gs = self.model.mu_params_out[2].cpu().data.numpy()
                    mu_infs = self.model.mu_params_out[3].cpu().data.numpy()

                    np.savetxt(ewpf, eps_wps, fmt='%.3f')
                    np.savetxt(ew0f, eps_w0s, fmt='%.3f')
                    np.savetxt(egf, eps_gs, fmt='%.3f')
                    np.savetxt(eif, eps_infs, fmt='%.3f')
                    np.savetxt(mwpf, mu_wps, fmt='%.3f')
                    np.savetxt(mw0f, mu_w0s, fmt='%.3f')
                    np.savetxt(mgf, mu_gs, fmt='%.3f')
                    np.savetxt(mif, mu_infs, fmt='%.3f')

        return eps_wp_file, eps_w0_file, eps_g_file, eps_inf_file, mu_wp_file,mu_w0_file,mu_g_file,mu_inf_file

    def evaluate_physics_output(self, save_dir='eval/'):
        """
        Evaluates model and calculates the physical quantities for a test dataset.
        :return: Eps, Mu, Eps_eff, Mu_eff, n_eff, and phase advance
        """
        self.load()                             # load the model as constructed
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        self.model.eval()                       # Evaluation mode

        # Get the file names
        eps_re_file = os.path.join(save_dir, 'test_eps_re_{}.csv'.format(self.saved_model))
        eps_im_file = os.path.join(save_dir, 'test_eps_im_{}.csv'.format(self.saved_model))
        mu_re_file = os.path.join(save_dir, 'test_mu_re_{}.csv'.format(self.saved_model))
        mu_im_file = os.path.join(save_dir, 'test_mu_im_{}.csv'.format(self.saved_model))
        eps_eff_re_file = os.path.join(save_dir, 'test_eps_eff_re_{}.csv'.format(self.saved_model))
        eps_eff_im_file = os.path.join(save_dir, 'test_eps_eff_im_{}.csv'.format(self.saved_model))
        mu_eff_re_file = os.path.join(save_dir, 'test_mu_eff_re_{}.csv'.format(self.saved_model))
        mu_eff_im_file = os.path.join(save_dir, 'test_mu_eff_im_{}.csv'.format(self.saved_model))
        n_eff_re_file = os.path.join(save_dir, 'test_n_eff_re_{}.csv'.format(self.saved_model))
        n_eff_im_file = os.path.join(save_dir, 'test_n_eff_im_{}.csv'.format(self.saved_model))
        theta_re_file = os.path.join(save_dir, 'test_theta_re_{}.csv'.format(self.saved_model))
        theta_im_file = os.path.join(save_dir, 'test_theta_im_{}.csv'.format(self.saved_model))
        adv_re_file = os.path.join(save_dir, 'test_adv_re_{}.csv'.format(self.saved_model))
        adv_im_file = os.path.join(save_dir, 'test_adv_im_{}.csv'.format(self.saved_model))

        # Open those files to append
        with open(eps_re_file, 'a') as fep_re, open(eps_im_file, 'a') as fep_im, \
            open(mu_re_file, 'a') as fmu_re, open(mu_im_file, 'a') as fmu_im, \
            open(eps_eff_re_file, 'a') as fepeff_re, open(eps_eff_im_file, 'a') as fepeff_im, \
            open(mu_eff_re_file, 'a') as fmueff_re, open(mu_eff_im_file, 'a') as fmueff_im, \
            open(n_eff_re_file, 'a') as fneff_re, open(n_eff_im_file, 'a') as fneff_im, \
            open(theta_re_file, 'a') as ftheta_re, open(theta_im_file, 'a') as ftheta_im, \
            open(adv_re_file, 'a') as fadv_re, open(adv_im_file, 'a') as fadv_im:

            # Loop through the eval data and evaluate
            with torch.no_grad():
                for ind, (geometry, spectra) in enumerate(self.test_loader):
                    if cuda:
                        geometry = geometry.cuda()
                        spectra = spectra.cuda()
                    pred_r, pred_t = self.model(geometry)  # Get the output

                    np.savetxt(fep_re, self.model.eps_out.real.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(fep_im, self.model.eps_out.imag.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(fmu_re, self.model.mu_out.real.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(fmu_im, self.model.mu_out.imag.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(fepeff_re, self.model.eps_eff_out.real.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(fepeff_im, self.model.eps_eff_out.imag.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(fmueff_re, self.model.mu_eff_out.real.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(fmueff_im, self.model.mu_eff_out.imag.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(fneff_re, self.model.n_out.real.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(fneff_im, self.model.n_out.imag.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(ftheta_re, self.model.theta_out.real.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(ftheta_im, self.model.theta_out.imag.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(fadv_re, self.model.adv_out.real.cpu().data.numpy(), fmt='%.3f')
                    np.savetxt(fadv_im, self.model.adv_out.imag.cpu().data.numpy(), fmt='%.3f')

        return eps_re_file, eps_im_file, mu_re_file, mu_im_file,\
               eps_eff_re_file, eps_eff_im_file, mu_eff_re_file, mu_eff_im_file, n_eff_re_file, n_eff_im_file, \
                theta_re_file, theta_im_file, adv_re_file, adv_im_file

    def record_weight(self, name='Weights', layer=None, batch=999, epoch=999):
        """
        Record the weights for a specific layer to tensorboard (save to file if desired)
        :input: name: The name to save
        :input: layer: The layer to check
        """
        if batch == 0:
            weights = layer.weight.cpu().data.numpy()   # Get the weights

            # if epoch == 0:
            # np.savetxt('Training_Weights_Lorentz_Layer' + name,
            #     weights, fmt='%.3f', delimiter=',')

            # Reshape the weights into a square dimension for plotting, zero padding if necessary
            wmin = np.amin(np.asarray(weights.shape))
            wmax = np.amax(np.asarray(weights.shape))
            sq = int(np.floor(np.sqrt(wmin * wmax)) + 1)
            diff = np.zeros((1, int(sq**2 - (wmin * wmax))), dtype='float64')
            weights = weights.reshape((1, -1))
            weights = np.concatenate((weights, diff), axis=1)
            f = plot_weights_3D(weights.reshape((sq, sq)), sq)
            self.log.add_figure(tag='1_Weights_' + name + '_Layer'.format(1),
                                figure=f, global_step=epoch)

    def record_grad(self, name='Gradients', layer=None, batch=999, epoch=999):
        """
        Record the gradients for a specific layer to tensorboard (save to file if desired)
        :input: name: The name to save
        :input: layer: The layer to check
        """
        if batch == 0 and epoch > 0:
            gradients = layer.weight.grad.cpu().data.numpy()

            # if epoch == 0:
            # np.savetxt('Training_Weights_Lorentz_Layer' + name,
            #     weights, fmt='%.3f', delimiter=',')

            # Reshape the weights into a square dimension for plotting, zero padding if necessary
            wmin = np.amin(np.asarray(gradients.shape))
            wmax = np.amax(np.asarray(gradients.shape))
            sq = int(np.floor(np.sqrt(wmin * wmax)) + 1)
            diff = np.zeros((1, int(sq ** 2 - (wmin * wmax))), dtype='float64')
            gradients = gradients.reshape((1, -1))
            gradients = np.concatenate((gradients, diff), axis=1)
            f = plot_weights_3D(gradients.reshape((sq, sq)), sq)
            self.log.add_figure(tag='1_Gradients_' + name + '_Layer'.format(1),
                                figure=f, global_step=epoch)

    def train(self):
        """
        The major training function. This starts the training using parameters given in the flags
        :return: None        """

        print("Starting training process")
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()

        # Construct optimizer after the model moved to GPU
        self.optm = self.make_optimizer()
        self.lr_scheduler = self.make_lr_scheduler()

        # Start a tensorboard session for logging loss and training images
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.ckpt_dir])
        url = tb.launch()
        print("TensorBoard started at %s" % url)

        for epoch in range(self.flags.train_step):

            if epoch == 1000:
                self.model.kill_osc = True

            # print("This is training Epoch {}".format(epoch))
            # Set to Training Mode
            train_loss = []
            train_loss_eval_mode_list = []
            self.model.train()

            for j, (geometry, spectra) in enumerate(self.train_loader):

                if cuda:
                    geometry = geometry.cuda()                          # Put data onto GPU
                    spectra = spectra.cuda()                            # Put data onto GPU

                self.optm.zero_grad()                                   # Zero the gradient first
                pred_r, pred_t = self.model(geometry)                   # Get the output (scattering parameters)

                loss = self.make_custom_loss(pred_r, pred_t, spectra)   # Calculate the loss

                if torch.isnan(loss) or torch.isinf(loss):              # Throw error if loss becomes NaN
                    print('Loss is invalid at iteration ', epoch)

                # Saves image of network architecture
                # if j == 0 and epoch == 0:
                #     im = make_dot(loss, params=dict(self.model.named_parameters())).render("Model Graph",
                #                                                                            format="png",
                #                                                                            directory=self.ckpt_dir)
                loss.backward()

                # Monitor sum of gradients
                # grads = [x.grad for x in self.model.parameters()]
                # grads_sum = torch.tensor([0], dtype=torch.float32).cuda()
                # for x in grads:
                #     grads_sum += torch.sum(x)
                # # grads_sum = torch.sum(grads_sum)
                # self.log.add_scalar('Grad sum', grads_sum, int(epoch*len(self.train_loader) + j))

                # Clip gradients to help with training if needed
                if self.flags.use_clip:
                    if self.flags.use_clip:
                        # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.flags.grad_clip)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.flags.grad_clip)

                # Record weights or gradients
                # if epoch % self.flags.record_step == 0:
                #     self.record_grad(name='eps_w0', layer=self.model.eps_w0, batch=j, epoch=epoch)
                #     self.record_grad(name='eps_g', layer=self.model.eps_g, batch=j, epoch=epoch)

                # At every recording step, plot k spectral predictions
                if epoch % self.flags.record_step == 0:
                    for b in range(1):
                        if j == b:
                            for k in range(self.flags.num_plot_compare):

                                # Two main plotting functions, either R/T or a full optical constants debugging graph

                                # f = plot_complex(logit1=square(abs(pred_t[k, :])).cpu().data.numpy(),
                                #                  tr1 = square(abs(spectra[k,:,1])).cpu().data.numpy(),
                                #                  logit2=square(abs(pred_r[k, :])).cpu().data.numpy(),
                                #                  tr2 = square(abs(spectra[k,:,0])).cpu().data.numpy(),
                                #                  xmin=self.flags.freq_low, xmax=self.flags.freq_high,
                                #                  num_points=self.flags.num_spec_points)
                                # self.log.add_figure(tag='Test ' + str(k) +') Sample T,R Spectra'.format(1),
                                #                     figure=f, global_step=epoch)

                                # f = plot_complex(logit1=pred_t[k, :].cpu().data.numpy(),
                                #                  tr1 = square(spectra[k,:,1].abs()).cpu().data.numpy(),
                                #                  logit2=spectra[k, :, 1].real.cpu().data.numpy(),
                                #                  tr2 = spectra[k, :, 1].imag.cpu().data.numpy(),
                                #                  xmin=self.flags.freq_low, xmax=self.flags.freq_high,
                                #                  num_points=self.flags.num_spec_points)
                                # self.log.add_figure(tag='Test ' + str(k) +') Sample Transmission Spectrum'.format(1),
                                #                     figure=f, global_step=epoch)

                                # logit1 = square(abs(pred_t)).cpu().data.numpy()
                                # tr1 = square(abs(spectra[:, :, 1])).cpu().data.numpy()
                                # logit2 = square(abs(pred_r)).cpu().data.numpy()
                                # tr2 = square(abs(spectra[:, :, 0])).cpu().data.numpy()

                                logit1 = pred_t.imag.cpu().data.numpy()
                                tr1 = spectra[:,:,1].imag.cpu().data.numpy()
                                logit2 = pred_r.imag.cpu().data.numpy()
                                tr2 = spectra[:,:,0].imag.cpu().data.numpy()

                                f = plot_all(logit1=logit1[k, :],tr1 = tr1[k, :], logit2=logit2[k, :],tr2 = tr2[k, :],
                                                 model=self.model, index=k, xmin=self.flags.freq_low,
                                                    xmax=self.flags.freq_high, num_points=self.flags.num_spec_points,
                                               num_osc=self.flags.num_lorentz_osc, y_axis='S11 (im), S21 (im)',
                                               label_y1='S21 (pred)', label_y2='S11 (pred)')
                                self.log.add_figure(tag='Test ' + str(k) + ' Batch ' + str(b) +
                                                        ' Plot Optical Constants'.format(1),
                                                    figure=f, global_step=epoch)

                self.optm.step()                                        # Move one step the optimizer
                train_loss.append(np.copy(loss.cpu().data.numpy()))     # Aggregate the loss

            # Calculate the avg loss of training
            train_avg_loss = np.mean(train_loss)

            if epoch % self.flags.eval_step == 0:           # For eval steps, do the evaluations and tensor board
                # Record the training loss to tensorboard
                self.log.add_scalar('Loss/ Training Loss', train_avg_loss, epoch)

                # Set to Evaluation Mode
                self.model.eval()
                print("Doing Evaluation on the model now")
                test_loss = []
                test_loss_mse = []         # In case main loss function is not MSE, monitor MSE as well
                with torch.no_grad():
                    for j, (geometry, spectra) in enumerate(self.test_loader):  # Loop through the eval set
                        if cuda:
                            geometry = geometry.cuda()
                            spectra = spectra.cuda()
                        pred_r, pred_t = self.model(geometry)  # Get the output
                        loss = self.make_custom_loss(pred_r, pred_t, spectra)
                        mse_loss = nn.functional.mse_loss(square(abs(pred_t)).float(),
                                                       square(abs(spectra[:, :, 1])).float(), reduction='mean')
                        test_loss.append(np.copy(loss.cpu().data.numpy()))           # Aggregate the loss
                        test_loss_mse.append(np.copy(mse_loss.cpu().data.numpy()))

                        # Monitor the MSE distribution for a test batch
                        # if j == 0 and epoch > 10 and epoch % self.flags.record_step == 0:
                        #     # f2 = plotMSELossDistrib(test_loss_mse)
                        #     f2 = plotMSELossDistrib(logit.cpu().data.numpy(), spectra[:, ].cpu().data.numpy())
                        #     self.log.add_figure(tag='0_Testing Loss Histogram'.format(1), figure=f2,
                        #                         global_step=epoch)

                # Record the testing loss to the tensorboard
                test_avg_loss = np.mean(test_loss)
                test_avg_loss_mse = np.mean(test_loss_mse)
                self.log.add_scalar('Loss/ Validation Loss', test_avg_loss, epoch)
                self.log.add_scalar('Loss/ MSE Loss', test_avg_loss_mse, epoch)

                print("This is Epoch %d, training loss %.5f, validation loss %.5f, mse loss %.5f" \
                      % (epoch, train_avg_loss, test_avg_loss, test_avg_loss_mse))

                # If model is improving, save the model
                if test_avg_loss < self.best_validation_loss:
                    self.best_validation_loss = test_avg_loss
                    self.best_mse_loss = test_avg_loss_mse
                    self.save()
                    print("Saving the model...")

                    if self.best_validation_loss < self.flags.stop_threshold:
                        print("Training finished EARLIER at epoch %d, reaching loss of %.5f" %\
                              (epoch, self.best_validation_loss))
                        return None

            # # Learning rate decay upon plateau
            self.lr_scheduler.step(train_avg_loss)
            # # self.lr_scheduler.step()

            # Handle warm restarts here. Can either use original lr or a smaller lr to reset to.
            if epoch > 10:
                # restart_lr = self.flags.lr * 0.01
                restart_lr = 5e-3
                if self.flags.use_warm_restart:
                    if epoch % self.flags.lr_warm_restart == 0:
                        for param_group in self.optm.param_groups:
                            param_group['lr'] = restart_lr
                            print('Resetting learning rate to %.5f' % restart_lr)

        # print('Finished')
        self.log.close()
