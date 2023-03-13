import pickle

from addict import Dict

from common import *

np.random.seed(0)

##################################################################
input_size = 784
output_size = 10
num_epochs = 20

resume_from = 0  # if 0, train from scratch, else resume from checkpoint epoch

##################################################################

config = {'activation': 'LeakyReLU', 'batch_size': 8, 'dropout': 0, 'hidden_size': 600,
          'learning_rate': 0.1997769342394881, 'loss': 'mse', 'momentum': 0.29799698061228563, 'num_layers': 4,
          'optimizer': 'SGD', 'use_bias': False}

config = Dict(config)
print(config)

# build model
if resume_from == 0:
    model = MLP(input_size=input_size,
                output_size=output_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                activation=get_activation(config.activation),
                use_bias=config.use_bias)
else:
    with open(f"best_e{resume_from}.pkl", "rb") as f:
        model = pickle.load(f)

# optimizer
optimizer = SGD(model, lr=config.learning_rate, momentum=config.momentum)
criterion = MSE()

train_loader = Dataloader(file_path='./mnist_train.csv', batch_size=config.batch_size)


def train():
    last_best_acc = 0
    # Train network
    for epoch in range(resume_from, num_epochs):
        # train for one epoch
        for inputs, label in tqdm(train_loader):
            output = model(inputs)

            # convert label to a distribution
            label = one_hot(label, num_classes=10)
            loss = criterion(output, label)
            # Backward and optimize
            loss_grad = criterion.backward()
            optimizer.step(loss_grad)

        # evaluate on validation set
        val_acc = test(model, test_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss:.8f}, '
              f'Val Accuracy: {float(val_acc):.8f}')
        if val_acc >= last_best_acc:
            last_best_acc = val_acc
            print("Saving model...")
            pickle.dump(model, open(f"best_e{epoch + 1}.pkl", "wb"))


train()
val_acc = test(model, test_loader)
print("performance = ", val_acc)
