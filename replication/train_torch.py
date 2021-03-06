# Model training and evaluation


from pyoneer_main.datagen import SimpleSequence
import replication.models_torch as models_torch
import replication.loss_torch as loss
from omegaconf import OmegaConf
import numpy as np
import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd


# comment this line out to use gpu:
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# %% Load and specify parameters

p = OmegaConf.load('params_noise.yml')

run_eagerly = True     # set to true to debug model training

# %% Data split parameters
# train-validation-test split

# semi-supervised learning:
if p.data_split == 'cifar10_ssl_default':
    data_split = {'trainIDs': range(49000), 'valIDs': range(49000, 50000),
                  'testIDs': range(50000, 60000)}
# supervised learning:
elif p.data_split == 'cifar10_default':
    data_split = {'trainIDs': range(4000), 'valIDs': range(49000, 50000),
                  'testIDs': range(50000, 60000)}
else:
    raise Exception('Data split not found: ', p.data_split)


# an indicator array indicating whether a training example is labeled
labeled = np.ones((60000, ), dtype = bool)

if p.data_split == 'cifar10_ssl_default':
    labeled[4000:49000, ...] = False



# Torch data loading
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
x_train = trainset.data
y_train = trainset.targets

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
x_test = testset.data
y_test = testset.targets

# %% Data load and prep

x = np.concatenate((x_train, x_test)).astype('float32')
y = np.concatenate((y_train, y_test))

# NOTE ADDED BECAUSE OF COMPATIBILITY ISSUES
# Swap x data
x = np.transpose(x, (0,3,2,1))

# class numbers -> one-hot representation
y = np.eye(y.max()+1)[y]

data = {'x': x, 'y': y, 'labeled': labeled}


def get_data_subset(data, split, subset):
    """
    Select training, validation or testing portion of the data.
    """
    # MODIFIED THIS AS WELL, split operation was not working
    return {arr: data[arr][split[subset + 'IDs'][0]:split[subset + 'IDs'][-1]+1] for arr in ['x', 'y', 'labeled']}

# %% Init generators

train_gen = SimpleSequence(p, data_split['trainIDs'],
                           data=get_data_subset(data, data_split, 'train'))

val_gen = SimpleSequence(p, data_split['valIDs'],
                         data=get_data_subset(data, data_split, 'val'))

test_gen = SimpleSequence(p, data_split['testIDs'],
                          data=get_data_subset(data, data_split, 'test'))


# %% Build the model architecture

arch = getattr(models_torch, p.arch.name)(torch.Tensor(3, 32, 32), **p.arch.params)

print(arch)


# create an optimizer
opt = torch.optim.Adam(arch.parameters(), lr=0.001)

# # create metrics
# metrics = [getattr(tf.keras.metrics, metric_class)(name=('%s_%s' % (metric_type, metric_name)))
#            for metric_type in ['sup', 'usup']
#            for metric_class, metric_name in zip(['CategoricalAccuracy'], ['acc'])]


model = models_torch.SemiSupervisedConsistencyModelTorch(arch)

if p.transform_output:
    p.transform_output = OmegaConf.merge(p.transform,
                                         {} if p.transform_output == True else p.transform_output)

# ----- Our training loop -------
start = time.time()

criterion = loss.custom_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

losses = pd.DataFrame()

# for t in range(0, p.epochs):
for t in range(0, 5):
    print('epoch ', t)
    for i in tqdm(range(0, train_gen.__len__())):
    # for i in tqdm(range(0, 30)):
        # obtain data from the generator
        inputs, y, labeled = train_gen.__getitem__(i)

        x = torch.Tensor(inputs[0])
        transform_parameters = torch.Tensor(inputs[1:])

        # x = torch.Tensor(x)
        y = torch.Tensor(y)
        labeled = torch.Tensor(labeled)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()

        x_flat = torch.flatten(x, start_dim=1)
        # forward pass
        # y_pred = model(x)

        # compute loss
        loss_total, loss_sup, loss_usup, (yl, predl), (pred1, pred2) = criterion(((x, transform_parameters), y, labeled), model, p)

        loss_total.backward()
        optimizer.step()
    train_gen.on_epoch_end()
    print(str(t) + "\n")
    print('epoch ', t, 'done. Loss:')
    losses = losses.append([loss_total.item(), loss_sup.item(), loss_usup.item()])
    print("loss: ", loss_total.item(), ", loss_sup: ", loss_sup.item(), ", loss_usup: ", loss_usup.item())

print('all losses:')
print(losses)
losses.to_csv('./losses.csv')

