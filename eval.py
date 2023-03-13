import pickle

from common import *


with open('model.pkl', "rb") as f:
    model = pickle.load(f)

val_acc = test(model, test_loader)
print("performance =", float(val_acc))
