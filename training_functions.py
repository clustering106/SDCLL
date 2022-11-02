import pandas as pd
from matplotlib import pyplot as plt
from torch import optim, nn
from tqdm import tqdm
import torch.nn.functional as F
import nets
import utils
import time
import torch
import numpy as np
import copy
from sklearn.cluster import KMeans


# Training function (from my torch_DCEC implementation, kept for completeness)
def train_model(model, dataloader, criteria, optimizers, schedulers, num_epochs, params):

    # Note the time
    since = time.time()
    # Unpack parameters
    writer = params['writer']
    if writer is not None: board = True
    txt_file = params['txt_file']
    pretrained = params['model_files'][1]
    pretrain = params['pretrain']
    print_freq = params['print_freq']
    dataset_size = params['dataset_size']
    device = params['device']
    batch = params['batch']
    pretrain_epochs = params['pretrain_epochs']
    gamma = params['gamma']
    update_interval = params['update_interval']
    tol = params['tol']

    dl = dataloader

    loss_module = nets.LossNet()
    optimizer1 = optim.Adam(loss_module.parameters())

    # Pretrain or load weights
    if pretrain:
        while True:
            pretrained_model = pretraining(model, copy.deepcopy(dl), criteria[0], optimizers[1], schedulers[1], pretrain_epochs, params)
            if pretrained_model:
                break
            else:
                for layer in model.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
        model = pretrained_model
    else:
        try:
            model.load_state_dict(torch.load(pretrained))
            utils.print_both(txt_file, 'Pretrained weights loaded from file: ' + str(pretrained))
        except:
            print("Couldn't load pretrained weights")

    # Initialise clusters
    utils.print_both(txt_file, '\nInitializing cluster centers based on K-means')
    kmeans(model, copy.deepcopy(dl), params)

    utils.print_both(txt_file, '\nBegin clusters training')

    # Prep variables for weights and accuracy of the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    # Initial target distribution
    utils.print_both(txt_file, '\nUpdating target distribution')
    output_distribution, labels, preds_prev = calculate_predictions(model, copy.deepcopy(dl), params, 0.00001)
    target_distribution = target(output_distribution)
    nmi = utils.metrics.nmi(labels, preds_prev)
    ari = utils.metrics.ari(labels, preds_prev)
    acc = utils.metrics.acc(labels, preds_prev)
    utils.print_both(txt_file,
                     'NMI: {0:.5f}\tARI: {1:.5f}\tAcc {2:.5f}\n'.format(nmi, ari, acc))

    if board:
        niter = 0
        writer.add_scalar('/NMI', nmi, niter)
        writer.add_scalar('/ARI', ari, niter)
        writer.add_scalar('/Acc', acc, niter)

    update_iter = 1
    finished = False
    # Go through all epochs
    l = 1
    for epoch in range(num_epochs):

        utils.print_both(txt_file, 'Epoch {}/{}'.format(epoch + 1, num_epochs))
        utils.print_both(txt_file,  '-' * 10)

        schedulers[0].step()
        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_loss_rec = 0.0
        running_loss_clust = 0.0

        # Keep the batch number for inter-phase statistics
        batch_num = 1
        img_counter = 0
        l = l * 0.00001
        # Iterate over data.
        for data in dataloader:
            # Get the inputs and labels
            inputs, _ = data
            inputs = inputs.to(device)
            label = []
            data1 = inputs
            output, feature1, feature2, feature3 = model(inputs)
            t = 0
            for i in range(len(feature1)):
                if (feature1[i].max() - find_max(feature1[i])) > l:
                    label.append(_[i])

                else:

                    data1 = del_tensor_ele(data1, i - t)
                    t += 1

            label = np.array(label)
            label = torch.tensor([item for item in label])


            # Uptade target distribution, chack and print performance
            if (batch_num - 1) % update_interval == 0 and not (batch_num == 1 and epoch == 0):
                utils.print_both(txt_file, '\nUpdating target distribution:')
                output_distribution, labels, preds = calculate_predictions(model, dataloader, params, l)
                target_distribution = target(output_distribution)

                nmi = utils.metrics.nmi(labels, preds)
                ari = utils.metrics.ari(labels, preds)
                acc = utils.metrics.acc(labels, preds)
                utils.print_both(txt_file,
                                 'NMI: {0:.5f}\tARI: {1:.5f}\tAcc {2:.5f}\t'.format(nmi, ari, acc))
                if board:
                    niter = update_iter
                    writer.add_scalar('/NMI', nmi, niter)
                    writer.add_scalar('/ARI', ari, niter)
                    writer.add_scalar('/Acc', acc, niter)
                    update_iter += 1

                # check stop criterion
                delta_label = np.sum(preds != preds_prev).astype(np.float32) / preds.shape[0]
                preds_prev = np.copy(preds)
                if delta_label < tol:
                    utils.print_both(txt_file, 'Label divergence ' + str(delta_label) + '< tol ' + str(tol))
                    utils.print_both(txt_file, 'Reached tolerance threshold. Stopping training.')
                    finished = True
                    break

            tar_dist = target_distribution[((batch_num - 1) * label.shape[0]):(batch_num*label.shape[0]), :]
            tar_dist = torch.from_numpy(tar_dist).to(device)
            # zero the parameter gradients
            optimizers[0].zero_grad()
            optimizer1.zero_grad()
            # Calculate losses and backpropagate
            with torch.set_grad_enabled(True):
                uncertainty = get_uncertainty(model, loss_module, data1)
                # Index in ascending order
                arg = np.argsort(uncertainty)
                target2 = []
                data2 = []
                # Update the labeled dataset and the unlabeled dataset, respectively
                data2 += list(torch.tensor(data1)[arg][:250].numpy())
                data2 = torch.tensor(data2)
                target2 += list(torch.tensor(tar_dist)[arg][:250].numpy())
                target2 = np.argmax(target2, axis=1)
                target2 = torch.tensor(target2)
                outputs, clusters, _, feature3 = model(data2)
                loss_pred = loss_module(feature3)
                print(loss_pred.shape)

                loss_target = nn.functional.cross_entropy(clusters, target2, reduction='none')
                print(loss_target.shape)
                loss2 = LossPredLoss(loss_pred, torch.log(loss_target))
                loss_rec = criteria[0](outputs, data2)
                loss_clust = 0.5*torch.sum(loss_target) / loss_target.size(0)
                loss = loss_rec + loss_clust + loss2*0.5
                loss.backward()
                optimizers[0].step()
                optimizer1.step()

            # For keeping statistics
            running_loss += loss.item() * inputs.size(0)
            running_loss_rec += loss_rec.item() * inputs.size(0)
            running_loss_clust += loss_clust.item() * inputs.size(0)

            # Some current stats
            loss_batch = loss.item()
            loss_batch_rec = loss_rec.item()
            loss_batch_clust = loss_clust.item()
            loss_accum = running_loss / ((batch_num - 1) * label.shape[0] + inputs.size(0))
            loss_accum_rec = running_loss_rec / ((batch_num - 1) * label.shape[0] + inputs.size(0))
            loss_accum_clust = running_loss_clust / ((batch_num - 1) * label.shape[0] + inputs.size(0))

            if batch_num % print_freq == 0:
                utils.print_both(txt_file, 'Epoch: [{0}][{1}/{2}]\t'
                                           'Loss {3:.4f} ({4:.4f})\t'
                                           'Loss_recovery {5:.4f} ({6:.4f})\t'
                                           'Loss clustering {7:.4f} ({8:.4f})\t'.format(epoch + 1, batch_num,
                                                                                        len(dataloader),
                                                                                        loss_batch,
                                                                                        loss_accum, loss_batch_rec,
                                                                                        loss_accum_rec,
                                                                                        loss_batch_clust,
                                                                                        loss_accum_clust))
                if board:
                    niter = epoch * len(dataloader) + batch_num
                    writer.add_scalar('/Loss', loss_accum, niter)
                    writer.add_scalar('/Loss_recovery', loss_accum_rec, niter)
                    writer.add_scalar('/Loss_clustering', loss_accum_clust, niter)
            batch_num = batch_num + 1

            # Print image to tensorboard
            if batch_num == len(dataloader) and (epoch+1) % 5:
                inp = utils.tensor2img(inputs)
                out = utils.tensor2img(outputs)
                if board:
                    img = np.concatenate((inp, out), axis=1)
                    writer.add_image('Clustering/Epoch_' + str(epoch + 1).zfill(3) + '/Sample_' + str(img_counter).zfill(2), img)
                    img_counter += 1

        if finished: break

        epoch_loss = running_loss / dataset_size
        epoch_loss_rec = running_loss_rec / dataset_size
        epoch_loss_clust = running_loss_clust / dataset_size

        if board:
            writer.add_scalar('/Loss' + '/Epoch', epoch_loss, epoch + 1)
            writer.add_scalar('/Loss_rec' + '/Epoch', epoch_loss_rec, epoch + 1)
            writer.add_scalar('/Loss_clust' + '/Epoch', epoch_loss_clust, epoch + 1)

        utils.print_both(txt_file, 'Loss: {0:.4f}\tLoss_recovery: {1:.4f}\tLoss_clustering: {2:.4f}'.format(epoch_loss,
                                                                                                            epoch_loss_rec,
                                                                                                           epoch_loss_clust))

        # If wanted to do some criterium in the future (for now useless)
        if epoch_loss < best_loss or epoch_loss >= best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        utils.print_both(txt_file, '')

    time_elapsed = time.time() - since
    utils.print_both(txt_file, 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Pretraining function for recovery loss only
def pretraining(model, dataloader, criterion, optimizer, scheduler, num_epochs, params):
    # Note the time
    since = time.time()

    # Unpack parameters
    writer = params['writer']
    if writer is not None: board = True
    txt_file = params['txt_file']
    pretrained = params['model_files'][1]
    print_freq = params['print_freq']
    dataset_size = params['dataset_size']
    device = params['device']
    batch = params['batch']

    # Prep variables for weights and accuracy of the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    # Go through all epochs
    for epoch in range(num_epochs):
        utils.print_both(txt_file, 'Pretraining:\tEpoch {}/{}'.format(epoch + 1, num_epochs))
        utils.print_both(txt_file, '-' * 10)

        scheduler.step()
        model.train(True)  # Set model to training mode

        running_loss = 0.0

        # Keep the batch number for inter-phase statistics
        batch_num = 1
        # Images to show
        img_counter = 0

        # Iterate over data.
        for data in dataloader:
            # Get the inputs and labels
            inputs, label = data
            inputs = inputs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs, _, outputs1, _ = model(inputs)
                loss = nn.functional.cross_entropy(outputs1,label)

                # loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

            # For keeping statistics
            running_loss += loss.item() * inputs.size(0)

            # Some current stats
            loss_batch = loss.item()
            loss_accum = running_loss / ((batch_num - 1) * batch + inputs.size(0))

            if batch_num % print_freq == 0:
                utils.print_both(txt_file, 'Pretraining:\tEpoch: [{0}][{1}/{2}]\t'
                           'Loss {3:.4f} ({4:.4f})\t'.format(epoch + 1, batch_num, len(dataloader),
                                                             loss_batch,
                                                             loss_accum))
                if board:
                    niter = epoch * len(dataloader) + batch_num
                    writer.add_scalar('Pretraining/Loss', loss_accum, niter)
            batch_num = batch_num + 1

            if batch_num in [len(dataloader), len(dataloader)//2, len(dataloader)//4, 3*len(dataloader)//4]:
                inp = utils.tensor2img(inputs)
                out = utils.tensor2img(outputs)
                if board:
                    img = np.concatenate((inp, out), axis=1)
                    writer.add_image('Pretraining/Epoch_' + str(epoch + 1).zfill(3) + '/Sample_' + str(img_counter).zfill(2), img)
                    img_counter += 1

        epoch_loss = running_loss / dataset_size
        if epoch == 0: first_loss = epoch_loss
        if epoch == 4 and epoch_loss / first_loss > 1:
            utils.print_both(txt_file, "\nLoss not converging, starting pretraining again\n")
            return False

        if board:
            writer.add_scalar('Pretraining/Loss' + '/Epoch', epoch_loss, epoch + 1)

        utils.print_both(txt_file, 'Pretraining:\t Loss: {:.4f}'.format(epoch_loss))

        # If wanted to add some criterium in the future
        if epoch_loss < best_loss or epoch_loss >= best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        utils.print_both(txt_file, '')

    time_elapsed = time.time() - since
    utils.print_both(txt_file, 'Pretraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    model.pretrained = True
    torch.save(model.state_dict(), pretrained)
    return model


# K-means clusters initialisation
def kmeans(model, dataloader, params):
    km = KMeans(n_clusters=model.num_clusters, n_init=20)
    output_array = None
    model.eval()
    # Itarate throught the data and concatenate the latent space representations of images
    for data in dataloader:
        inputs, _ = data
        inputs = inputs.to(params['device'])
        _, _, outputs, _ = model(inputs)

        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
        if output_array.shape[0] > 50000: break

    # Perform K-means
    km.fit_predict(output_array)
    # Update clustering layer weights
    weights = torch.from_numpy(km.cluster_centers_)
    model.clustering.set_weight(weights.to(params['device']))
    # torch.cuda.empty_cache()


# Function forwarding data through network, collecting clustering weight output and returning prediciotns and labels
def calculate_predictions(model, dataloader, params, l):
    output_array = None
    label_array = None
    model.eval()
    for data in dataloader:
        inputs, labels = data
        label = []
        data1 = inputs
        output, feature1, feature2,_ = model(inputs)

        t = 0

        for i in range(len(feature1)):
            if (feature1[i].max() - find_max(feature1[i])) > l:
                label.append(labels[i])

            else:
                data1 = del_tensor_ele(data1, i - t)
                t += 1

        label = np.array(label)
        label = torch.tensor([item for item in label])
        data1 = data1.to(params['device'])
        label = label.to(params['device'])
        _, outputs, _ ,_= model(data1)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
            label_array = np.concatenate((label_array, label.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
            label_array = label.cpu().detach().numpy()

    preds = np.argmax(output_array.data, axis=1)
    return output_array, label_array, preds


# Calculate target distribution
def target(out_distr):
    tar_dist = out_distr ** 2 / np.sum(out_distr, axis=0)
    tar_dist = np.transpose(np.transpose(tar_dist) / np.sum(tar_dist, axis=1))
    return tar_dist

def find_max(a):
    a = a.detach().numpy()
    b = np.zeros(10, dtype=int)
    c = np.zeros(10)
    c[0] = a.max()
    b[0] = np.where(a == c[0])[0][0]
    new_a = np.delete(a, b[0])
    c[1] = np.max(new_a)
    return c[1]

def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)


def get_uncertainty(models1, models2, data1):
    models1.eval()
    models2.eval()
    uncertainty = torch.tensor([])

    with torch.no_grad():
        inputs = data1
        x, out1, out2, features = models1(inputs)
        pred_loss = models2(features)
        pred_loss = pred_loss.view(pred_loss.size(0))
        uncertainty = torch.cat((uncertainty, pred_loss), 0)
    return uncertainty.cpu()

def LossPredLoss(input, target, margin=1.0,reduction='mean'):
    assert input.shape == input.flip(0).shape
    input = (input - input.flip(0))[
            :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()
    one = 2 * torch.sign(
        torch.clamp(target, min=0)) - 1
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)  # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss