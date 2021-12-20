from utils import create_data_iter, Batch
from train import load_model, train
from config import Config

config = Config()

model, optimizer, sheduler, device = load_model(config)
train_iter, dev_iter, test_iter = create_data_iter(config)
batch = Batch(config)
train(model, train_iter, dev_iter, optimizer, config, batch)
'''
pipreqs . --encoding=utf8 --force

'''