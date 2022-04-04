import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
sys.path.append(os.getcwd() + '/Swin_NER')

import math
import json
import torch
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
from Models.models import WrapperModel, BertTransformerCrf, BertSwinCrf, BertSwinTreeCrf

pl.seed_everything(4)


class Config:
    def __init__(self):
        # controller
        self.debug = True
        self.preprocess = False
        self.en_train = True
        self.en_inference = False
        self.inference_model_path = None
        self.model = BertSwinCrf

        # hardware
        self.accelerator = 'gpu'
        self.num_processes = 48 if self.accelerator == 'gpu' else 12
        self.gpus = 1 if self.accelerator == 'gpu' else 1
        self.devices = [2] if self.gpus == 1 else self.gpus
        self.strategy = 'dp' if (self.accelerator == 'gpu' and self.gpus > 1) else None
        self.precision = 32 if self.accelerator == 'gpu' else 32

        # utils
        self.log_dir = 'Results/logs/'
        self.model_dir = 'Results/models/'
        self.inference_result_path = 'Results/inference_result.json'

        # ner data
        self.ner_tagging = 'BIO'
        self.label_list = ['address', 'book', 'company', 'game', 'government', 'movie',
                            'name', 'organization', 'position', 'scene']
        self.num_types = len(self.label_list)
        self.num_tags = (4 * self.num_types + 1) if self.ner_tagging == 'BIOES' else (
            (2 * self.num_types + 1) if self.ner_tagging == 'BIO' else None)
        self.label2idx, self.idx2label = None, None

        # model
        self.batch_size = 16
        self.total_steps = 0
        self.plm_name = "bert-base-chinese"
        self.hidden_size = 768
        self.num_heads = 8
        self.num_layers = 4
        self.dropout = 0.3
        self.shift_size = 'cycle'
        self.model_args = {
            "reprocess_input_data": True,
            "max_seq_length": 64,
            "train_batch_size": self.batch_size * self.gpus,
            "num_train_epochs": 40,
            "save_eval_checkpoints": False,
            "save_model_every_epoch": False,
            "evaluate_during_training": True,
            "evaluate_generated_text": True,
            "evaluate_during_training_verbose": True,
            "use_multiprocessing": False,
            "manual_seed": 4,
            "save_steps": 11898,
            "gradient_accumulation_steps": 1,
            "early_stopping_patience": 8,
        }

        # fit
        # self.auto_scale_batch_size = "binsearch"


def _load_model_args(args):
    loaded_args = Seq2SeqArgs()
    loaded_args.update_from_dict(args)
    return loaded_args


if __name__ == '__main__':
    config = Config()
    config.model_args = _load_model_args(config.model_args)

    logger = TensorBoardLogger(
        save_dir=config.log_dir,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",
        mode="min",
        dirpath=config.model_dir,
        filename="swinclue-{epoch:02d}-{val_loss:.2f}",
    )

    trainer = Trainer(
        fast_dev_run=config.debug,
        max_epochs=config.model_args.num_train_epochs,
        accelerator=config.accelerator,
        gpus=config.gpus,
        strategy=config.strategy,
        logger=logger,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val_f1", patience=config.model_args.early_stopping_patience)],
        precision=config.precision,
        gradient_clip_val=config.model_args.max_grad_norm,
        devices=config.devices
    )

    data_module = NERDataModule(config)
    config.label2idx, config.idx2label = data_module.label2idx, data_module.idx2label

    if config.preprocess:
        data_module.my_prepare_data()

    if config.en_train:
        data_module.setup(stage='fit')
        # update training args
        tb_size = config.model_args.train_batch_size * max(1, config.gpus)
        ab_size = float(config.model_args.num_train_epochs) // trainer.accumulate_grad_batches
        config.total_steps = (data_module.train_len // tb_size) * ab_size
        config.model_args.warmup_steps = math.ceil(config.total_steps * config.model_args.warmup_ratio)

        model = WrapperModel(config)
        trainer.fit(model, data_module)

        output_json = {
            "best_model_path": checkpoint_callback.best_model_path,
        }
        print(output_json)