#%load_ext autoreload
#%autoreload 2
import os
import joblib
import sys
import time
sys.path.append("../GAN/")

from braindecode.datautil.iterators import get_balanced_batches
from eeggan.examples.conv_cub.model import Generator,Discriminator
from eeggan.util import weight_filler
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import random

plt.switch_backend('agg')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark=True

n_critic = 5
n_batch = 64
input_length = 768
jobid = 0

n_z = 200
n_chans = 22
lr = 0.001
n_blocks = 6
rampup = 2000.
block_epochs = [2000,4000,4000,4000,4000,4000]

task_ind = 0#subj_ind

np.random.seed(task_ind)
torch.manual_seed(task_ind)
torch.cuda.manual_seed_all(task_ind)
random.seed(task_ind)
rng = np.random.RandomState(task_ind)

training_file = '../../Project_Data/X_train_valid.npy'
training_file_targets = '../../Project_Data/y_train_valid.npy'
testing_file = '../../Project_Data/X_test.npy'
testing_file_targets = '../../Project_Data/y_test.npy'
EEG_train_data = np.load(training_file)
EEG_train_targets = np.load(training_file_targets)
EEG_test_data = np.load(testing_file)
EEG_test_targets = np.load(testing_file_targets)

train = np.concatenate((EEG_train_data,EEG_test_data))
target = np.concatenate((EEG_train_targets,EEG_test_targets))

train = train[:,:22,:,None]
train = train-train.mean()
train = train/train.std()
train = train/np.abs(train).max()
target_onehot = np.zeros((target.shape[0],4))
target_onehot[np.arange(len(target_onehot)),target - 769] = 1
print(train.shape)
print(target_onehot.shape)


modelpath = '../../Project_Data/models/GAN_debug/%s/'%('WGAN_adaptlambclamp_CONV_CUB_10l_run%d'%task_ind)
modelname = 'Progressive'
if not os.path.exists(modelpath):
    os.makedirs(modelpath)


generator = Generator(n_chans,n_z)
discriminator = Discriminator(n_chans)

generator.train_init(alpha=lr,betas=(0.,0.99))
discriminator.train_init(alpha=lr,betas=(0.,0.99),eps_center=0.001,
                        one_sided_penalty=True,distance_weighting=True)
generator = generator.apply(weight_filler)
discriminator = discriminator.apply(weight_filler)

i_block_tmp = 0
i_epoch_tmp = 0
generator.model.cur_block = i_block_tmp
discriminator.model.cur_block = n_blocks-1-i_block_tmp
fade_alpha = 1.
generator.model.alpha = fade_alpha
discriminator.model.alpha = fade_alpha

generator = generator.cuda()
discriminator = discriminator.cuda()
generator.train()
discriminator.train()

losses_d = []
losses_g = []
i_epoch = 0
z_vars_im = rng.normal(0,1,size=(1000,n_z)).astype(np.float32)

for i_block in range(i_block_tmp,n_blocks):
    c = 0
    torch.set_grad_enabled(False)
    train_tmp = discriminator.model.downsample_to_block(Variable(torch.from_numpy(train).cuda()),discriminator.model.cur_block).data.cpu()
    torch.set_grad_enabled(True)
    for i_epoch in range(i_epoch_tmp,block_epochs[i_block]):
        start = time.time()
        i_epoch_tmp = 0

        if fade_alpha<1:
            fade_alpha += 1./rampup
            generator.model.alpha = fade_alpha
            discriminator.model.alpha = fade_alpha

        batches = get_balanced_batches(train.shape[0], rng, True, batch_size=n_batch)
        iters = int(len(batches)/n_critic)

        for it in range(iters):
            for i_critic in range(n_critic):
                train_batches = train_tmp[batches[it*n_critic+i_critic]]
                batch_real = Variable(train_batches,requires_grad=True).cuda()
                print(len(batches[it*n_critic+i_critic]),n_z)
                z_vars = rng.normal(0,1,size=(len(batches[it*n_critic+i_critic]),n_z)).astype(np.float32)
                print(z_vars.shape)
                torch.set_grad_enabled(False)
                z_vars = Variable(torch.from_numpy(z_vars)).cuda()
                print(z_vars.shape)
                torch.set_grad_enabled(True)
                batch_fake = Variable(generator(z_vars).data,requires_grad=True).cuda()
                print(batch_real.shape,batch_fake.shape)
                loss_d = discriminator.train_batch(batch_real,batch_fake)
                assert np.all(np.isfinite(loss_d))
            z_vars = rng.normal(0,1,size=(n_batch,n_z)).astype(np.float32)
            z_vars = Variable(torch.from_numpy(z_vars),requires_grad=True).cuda()
            loss_g = generator.train_batch(z_vars,discriminator)

        losses_d.append(loss_d)
        losses_g.append(loss_g)


        if i_epoch%100 == 0:
            generator.eval()
            discriminator.eval()

            print('Epoch: %d   Loss_F: %.3f   Loss_R: %.3f   Penalty: %.4f   Loss_G: %.3f'%(i_epoch,loss_d[0],loss_d[1],loss_d[2],loss_g))
            joblib.dump((i_epoch,losses_d,losses_g),os.path.join(modelpath,modelname%jobid+'_.data'),compress=True)
            joblib.dump((i_epoch,losses_d,losses_g),os.path.join(modelpath,modelname%jobid+'_%d.data'%i_epoch),compress=True)
            #joblib.dump((n_epochs,n_z,n_critic,batch_size,lr),os.path.join(modelpath,modelname%jobid+'_%d.params'%i_epoch),compress=True)

            freqs_tmp = np.fft.rfftfreq(train_tmp.numpy().shape[2],d=1/(250./np.power(2,n_blocks-1-i_block)))

            train_fft = np.fft.rfft(train_tmp.numpy(),axis=2)
            train_amps = np.abs(train_fft).mean(axis=3).mean(axis=0).squeeze()


            z_vars = Variable(torch.from_numpy(z_vars_im),volatile=True).cuda()
            batch_fake = generator(z_vars)
            fake_fft = np.fft.rfft(batch_fake.data.cpu().numpy(),axis=2)
            fake_amps = np.abs(fake_fft).mean(axis=3).mean(axis=0).squeeze()

            plt.figure()
            plt.plot(freqs_tmp,np.log(fake_amps),label='Fake')
            plt.plot(freqs_tmp,np.log(train_amps),label='Real')
            plt.title('Frequency Spectrum')
            plt.xlabel('Hz')
            plt.legend()
            plt.savefig(os.path.join(modelpath,modelname%jobid+'_fft_%d_%d.png'%(i_block,i_epoch)))
            plt.close()

            batch_fake = batch_fake.data.cpu().numpy()
            plt.figure(figsize=(10,10))
            for i in range(10):
                plt.subplot(10,1,i+1)
                plt.plot(batch_fake[i].squeeze())
                plt.xticks((),())
                plt.yticks((),())
            plt.subplots_adjust(hspace=0)
            plt.savefig(os.path.join(modelpath,modelname%jobid+'_fakes_%d_%d.png'%(i_block,i_epoch)))
            plt.close()

            discriminator.save_model(os.path.join(modelpath,modelname%jobid+'.disc'))
            generator.save_model(os.path.join(modelpath,modelname%jobid+'.gen'))

            plt.figure(figsize=(10,15))
            plt.subplot(3,2,1)
            plt.plot(np.asarray(losses_d)[:,0],label='Loss Real')
            plt.plot(np.asarray(losses_d)[:,1],label='Loss Fake')
            plt.title('Losses Discriminator')
            plt.legend()
            plt.subplot(3,2,2)
            plt.plot(np.asarray(losses_d)[:,0]+np.asarray(losses_d)[:,1]+np.asarray(losses_d)[:,2],label='Loss')
            plt.title('Loss Discriminator')
            plt.legend()
            plt.subplot(3,2,3)
            plt.plot(np.asarray(losses_d)[:,2],label='Penalty Loss')
            plt.title('Penalty')
            plt.legend()
            plt.subplot(3,2,4)
            plt.plot(-np.asarray(losses_d)[:,0]-np.asarray(losses_d)[:,1],label='Wasserstein Distance')
            plt.title('Wasserstein Distance')
            plt.legend()
            plt.subplot(3,2,5)
            plt.plot(np.asarray(losses_g),label='Loss Generator')
            plt.title('Loss Generator')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(modelpath,modelname%jobid+'_losses.png'))
            plt.close()

            generator.train()
            discriminator.train()
        end = time.time()
        print("Epoch %d took %.3f seconds!"%(i_epoch,end - start))

    fade_alpha = 0.
    generator.model.cur_block += 1
    discriminator.model.cur_block -= 1
