from Models.models import BertTransformerCrf, BertSwinCrf, BertSwinTreeCrf, BertSwinMlpCrf

class Config:
    def __init__(self, args):
        # controller
        self.debug = False
        self.preprocess = False
        self.en_train = True
        self.en_test = False
        self.en_inference = False
        self.inference_model_path = None
        self.model = BertSwinMlpCrf  # BertTransformerCrf, BertSwinCrf, BertSwinTreeCrf, BertSwinMlpCrf
        self.shift_size = 'cycle'  # 'auto', 'cycle', and None
        # swin+mlp
        self.mlp_target = 'is_entity'  # is_entity, is_edge

        # hardware
        self.accelerator = 'gpu'
        self.num_processes = 48 if self.accelerator == 'gpu' else 12
        self.gpus = 1 if self.accelerator == 'gpu' and args.gpu_num == 0 else args.gpu_num
        self.device = args.gpu_device
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
        self.batch_size = args.batch_size
        self.total_steps = 0
        self.plm_name = "bert-base-chinese"
        self.hidden_size = 768
        self.num_heads = 8
        self.num_layers = 4
        self.dropout = 0.3
        self.model_args = {
            "reprocess_input_data": True,
            "max_seq_length": 64,
            "train_batch_size": self.batch_size * max(self.gpus, 1),
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