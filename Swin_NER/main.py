import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
sys.path.append(os.getcwd() + '/Swin_NER')

import math
import json
import torch
import argparse
from tqdm import tqdm
import pytorch_lightning as pl
from collections import defaultdict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from simpletransformers.config.model_args import Seq2SeqArgs
from transformers import BertTokenizer

from data_process import NERDataModule
from ner_metrics import *
from Models.models import WrapperModel, MlpWrapperModel
from config import Config

pl.seed_everything(4)

def parse():
    parser = argparse.ArgumentParser(description="swin4ner")    
    parser.add_argument('-batch_size', type=int, default=16, help='batch size')
    parser.add_argument('-gpu_num',type=int,default=0,help='used gpu number')
    parser.add_argument('-gpu_device',type=list,default=['0'],help='specify device')
    args = parser.parse_args()
    
    args.gpu_device[0] = int(args.gpu_device[0])
    return args

def _load_model_args(args):
    loaded_args = Seq2SeqArgs()
    loaded_args.update_from_dict(args)
    return loaded_args


if __name__ == '__main__':
    args = parse()
    config = Config(args)
    config.model_args = _load_model_args(config.model_args)

    data_module = NERDataModule(config)
    config.label2idx, config.idx2label = data_module.label2idx, data_module.idx2label

    if config.preprocess:
        data_module.my_prepare_data()

    if config.en_train or config.en_test or config.en_inference:
        logger = TensorBoardLogger(
            save_dir=config.log_dir,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_f1",
            mode="max",
            dirpath=config.model_dir,
            filename="swinclue-{epoch:02d}-{val_loss:.2f}-{val_f1:.2f}",
        )

        trainer = Trainer(
            fast_dev_run=config.debug,
            max_epochs=config.model_args.num_train_epochs,
            accelerator=config.accelerator,
            gpus=config.gpus,
            strategy=config.strategy,
            logger=logger,
            callbacks=[checkpoint_callback, EarlyStopping(monitor="val_f1", patience=config.model_args.early_stopping_patience,
                                                          mode="max")],
            precision=config.precision,
            gradient_clip_val=config.model_args.max_grad_norm,
            devices=config.device
        )

        if config.en_train:
            data_module.setup(stage='fit')
            # update training args
            tb_size = config.model_args.train_batch_size * max(1, config.gpus)
            ab_size = float(config.model_args.num_train_epochs) // trainer.accumulate_grad_batches
            config.total_steps = (data_module.train_len // tb_size) * ab_size
            config.model_args.warmup_steps = math.ceil(config.total_steps * config.model_args.warmup_ratio)

            model = MlpWrapperModel(config)
            trainer.fit(model, data_module)

            output_json = {
                "best_model_path": checkpoint_callback.best_model_path,
            }
            print(output_json)