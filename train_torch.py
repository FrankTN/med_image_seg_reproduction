# Model training and evaluation


from pyoneer_main.datagen import SimpleSequence
import pyoneer_main.func as func
import model_arch
from omegaconf import OmegaConf
import tensorflow as tf
import numpy as np
import os
import time
import torch
import torchvision
import torchvision.transforms as transforms


# comment this line out to use gpu:
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# %% Load and specify parameters

p = OmegaConf.load('params.yml')

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# %% Data load and prep

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x = np.concatenate((x_train, x_test)).astype('float32')
y = np.concatenate((y_train, y_test))

# NOTE ADDED BECAUSE OF COMPATIBILITY ISSUES
# Swap x data
x = tf.transpose(x,[0,3,2,1])

# class numbers -> one-hot representation
y = tf.keras.utils.to_categorical(y)

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

arch = getattr(model_arch, p.arch.name)(torch.Tensor(3, 32, 32))#**p.arch.params)

print(arch)


# create an optimizer
opt = torch.optim.Adam(arch.parameters(), lr=0.001)

# create metrics
metrics = [getattr(tf.keras.metrics, metric_class)(name=('%s_%s' % (metric_type, metric_name)))
           for metric_type in ['sup', 'usup']
           for metric_class, metric_name in zip(['CategoricalAccuracy'], ['acc'])]


# model = model_arch.SemiSupervisedConsistencyModel(inputs=[model_arch.input],
#                                               outputs=[model_arch.output])
# model.compile(optimizer = opt, loss = getattr(func, p.loss),
#               metrics = metrics, run_eagerly = run_eagerly, p = p)

data_np = np.array(train_gen.data)
data_torch = torch.from_numpy(data_np)
model = model_arch.SemiSupervisedConsistencyModelTorch(data_torch[0])

# %% Train the model

start = time.time()

history = model.fit(x = train_gen,
                    epochs = p.epochs,
                    verbose = 1,
                    validation_data = val_gen)

print('Training time: %.1f seconds.' % (time.time() - start))

# %% Evaluate the model on the test set

metric_values = model.evaluate(test_gen)

for metric_name, metric_value in zip(model.metrics_names, metric_values):
    print('%s: %.3f' % (metric_name, metric_value))


# ----- Our training loop -------
# x = torch.linspace(-math.pi, math.pi, 2000)
# y = torch.sin(x)
#
# model = SemiSupervisedConsistencyModelTorch()
#
# criterion = custom_loss
# optimizer = torch.optim.Adam()
# epochs = 1000

# for t in range(0, epochs):
#     # forward pass
#     y_pred = model(x)
#
#     # compute loss
#     loss = criterion(y_pred, y)
#     if t % 10 == 9:
#         print(t, loss.item())
#
#     # Zero gradients, perform a backward pass, and update the weights.
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
