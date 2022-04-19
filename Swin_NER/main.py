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

from config import Config

pl.seed_everything(4)

def parse():
    parser = argparse.ArgumentParser(description="swin4ner")    
    parser.add_argument('-debug', type=str, default='True', help='debug mode')
    parser.add_argument('-wrapper_class', type=str, default='MlpWrapperModel', help='wrapper class')
    parser.add_argument('-model', type=str, default='BertSwinMlpCrf', help='model class')
    parser.add_argument('-batch_size', type=int, default=2, help='batch size')
    parser.add_argument('-gpu_num',type=int,default=1,help='used gpu number')
    parser.add_argument('-gpu_device',type=str,default='0',help='specify device')
    parser.add_argument('-mlp_target',type=str,default='is_entity',help='mlp target')
    args = parser.parse_args()

    return args

def _load_model_args(args):
    loaded_args = Seq2SeqArgs()
    loaded_args.update_from_dict(args)
    return loaded_args


if __name__ == '__main__':
    args = parse()
    config = Config(args)
    config.model_args = _load_model_args(config.model_args)
    print(config)

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
            gradient_clip_val=config.model_args.max_grad_norm
        )

        if config.en_train:
            data_module.setup(stage='fit')
            # update training args
            tb_size = config.model_args.train_batch_size * max(1, config.gpus)
            ab_size = float(config.model_args.num_train_epochs) // trainer.accumulate_grad_batches
            config.total_steps = (data_module.train_len // tb_size) * ab_size
            config.model_args.warmup_steps = math.ceil(config.total_steps * config.model_args.warmup_ratio)

            model = config.wrapper_class(config)
            trainer.fit(model, data_module)

            output_json = {
                "best_model_path": checkpoint_callback.best_model_path,
            }
            print(output_json)