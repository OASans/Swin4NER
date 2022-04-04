import torch
from torch import nn
from pytorch_lightning import LightningModule
from transformers import (
    AdamW,
    BertConfig,
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from ner_metrics import *
from Models.crf import Crf
from Models.fusions import OriginTransformerEncoder, SwinTransformer, SwinTreeTransformer


class BertTransformerCrf(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bert = BertModel.from_pretrained(config.plm_name)
        self.fusion = OriginTransformerEncoder(config)
        self.crf = Crf(config)
    
    def forward(self, batch):
        output = self.bert(batch['input_ids'], batch['attention_mask'])['last_hidden_state']
        # batch['attention_mask'][:, 0] = 0
        output = self.fusion(output, batch['attention_mask'])
        output = self.crf(output, batch['attention_mask'])
        return output


class BertSwinCrf(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bert = BertModel.from_pretrained(config.plm_name)
        self.fusion = SwinTransformer(config)
        self.crf = Crf(config)
    
    def forward(self, batch):
        output = self.bert(batch['input_ids'], batch['attention_mask'])['last_hidden_state']
        # batch['attention_mask'][:, 0] = 0
        output = self.fusion(output, batch['attention_mask'])
        output = self.crf(output, batch['attention_mask'])
        return output


class BertSwinTreeCrf(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bert = BertModel.from_pretrained(config.plm_name)
        self.fusion = SwinTreeTransformer(config)
        self.crf = Crf(config)
    
    def forward(self, batch):
        output = self.bert(batch['input_ids'], batch['attention_mask'])['last_hidden_state']
        # batch['attention_mask'][:, 0] = 0
        output = self.fusion(output, batch['attention_mask'])
        output = self.crf(output, batch['attention_mask'])
        return output



class WrapperModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.args = config.model_args
        self.label2idx, self.idx2label = config.label2idx, config.idx2label
        self.total_steps = config.total_steps

        self.model = config.model(config)
        
    def forward(self, batch):
        output = self.model(batch)
        return output
    
    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        loss = self.model.crf.cal_loss(output, batch['y_true_bio'], batch['attention_mask'])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        output = self.model(batch)
        val_loss = self.model.crf.cal_loss(output, batch['y_true_bio'], batch['attention_mask'])

        return {'val_loss': val_loss, 'pred': torch.tensor(output['pred'], device=val_loss.get_device()), 'true': batch['y_true_bio']}
    
    def validation_step_end(self, batch_parts):
        val_loss = torch.mean(batch_parts['val_loss']) 
        return {'val_loss': val_loss, 'pred': batch_parts['pred'], 'true': batch_parts['true']}
    
    def validation_epoch_end(self, outputs) -> None:
        val_loss, pred_list, true_list = 0, [], []
        for output in outputs:
            val_loss += output['val_loss']
            pred_list.extend(output['pred'].tolist())
            true_list.extend(output['true'].tolist())

        val_loss /= len(outputs)
        pred_bio = [[self.idx2label[p] for p in pred] for pred in pred_list]
        true_bio = [[self.idx2label[p] for p in pred] for pred in true_list]
        pred_entities, true_entities = get_entities(pred_bio), get_entities(true_bio)
        
        self.log("val_loss", val_loss) 
        self.log("val_precision", precision_score(true_entities, pred_entities))
        self.log("val_recall", recall_score(true_entities, pred_entities))
        self.log("val_f1", f1_score(true_entities, pred_entities))
        print(classification_report(true_entities, pred_entities))
    
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [p for n, p in self.model.named_parameters() if n in params]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in self.model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in self.model.named_parameters()
                            if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in self.model.named_parameters()
                            if n not in custom_parameter_names and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

