import sys
sys.path.append("/kaggle/input/mamba-files/mamba-main")
sys.path.append("/kaggle/input/mamba-files/mamba-main/classification")
sys.path.append("/kaggle/input/mamba-files/mamba-main/classification/evalcap")
sys.path.append("/kaggle/input/mamba-files/mamba-main/classification/models")
sys.path.append("/kaggle/input/mamba-files/mamba-main/classification/models/mamba2")

from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer, AutoImageProcessor
from classification.models.vmamba2 import vmamba_tiny_m2, vmamba_small_m2, vmamba_base_m2
import json

import torch.nn as nn
import lightning.pytorch as pl
from einops import rearrange
from classification.evalcap.bleu.bleu import Bleu
from classification.evalcap.rouge.rouge import Rouge
from classification.evalcap.cider.cider import Cider
from classification.evalcap.meteor.meteor import Meteor

from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from torchvision.transforms import ToTensor, Compose, Resize
import os
from pprint import pprint
from lightning.pytorch import seed_everything
import torch
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import FSDPStrategy
import numpy as np
from IPython.display import FileLink, display
import time

import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = "/kaggle/input/serviceaccount/mamba-435814-24c0433c4b81.json"
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency = 100,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    # def on_train_batch_end(self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx):
    #     """ Check if we should save a checkpoint after every train batch """
    #     epoch = trainer.current_epoch
    #     global_step = trainer.global_step
    #     # print(f"I reached here global step: {global_step}")
    #     #print(epoch,global_step)
    #     if global_step % self.save_step_frequency == 0:
    #         if self.use_modelcheckpoint_filename:
    #             filename = trainer.checkpoint_callback.filename
    #         else:
    #             filename = f"{self.prefix}_{epoch=}_{global_step=}_{batch_idx=}.ckpt"
    #         ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
    #         trainer.save_checkpoint(ckpt_path)
    #         upload_file(ckpt_path,parent_folder_id='174nCSwFpvBgzKedFvz0MQ6CEtns79ZFh')
            
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        if self.use_modelcheckpoint_filename:
            filename = trainer.checkpoint_callback.filename
        else:
            filename = f"{self.prefix}_{epoch=}.ckpt"
        ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
        trainer.save_checkpoint(ckpt_path)
        upload_file(ckpt_path,parent_folder_id='174nCSwFpvBgzKedFvz0MQ6CEtns79ZFh')
            
def list_folder(parent_folder_id=None, delete=False):
    """List folders and files in Google Drive."""
    results = drive_service.files().list(
        q=f"'{parent_folder_id}' in parents and trashed=false" if parent_folder_id else None,
        pageSize=1000,
        fields="nextPageToken, files(id, name, mimeType)"
    ).execute()
    items = results.get('files', [])

    if not items:
        print("No folders or files found in Google Drive.")
    else:
        print("Folders and files in Google Drive:")
        for item in items:
            print(f"Name: {item['name']}, ID: {item['id']}, Type: {item['mimeType']}")
            if delete:
                delete_files(item['id'])
                
def delete_files(file_or_folder_id):
    """Delete a file or folder in Google Drive by ID."""
    try:
        drive_service.files().delete(fileId=file_or_folder_id).execute()
        print(f"Successfully deleted file/folder with ID: {file_or_folder_id}")
    except Exception as e:
        print(f"Error deleting file/folder with ID: {file_or_folder_id}")
        print(f"Error details: {str(e)}")

def upload_file(file_path, parent_folder_id=None,retry=5):
    """Upload a file to Google Drive."""
    try:
        file_metadata = {
            'name': os.path.basename(file_path),
            'parents': [parent_folder_id] if parent_folder_id else []
        }
        
        media = MediaFileUpload(file_path, resumable=True)
        
        created_file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id',
            supportsAllDrives=True
        ).execute()
        
        print(f'\nUploaded File ID: {created_file["id"]}', end='')
        return created_file["id"]
    except Exception as e:
        if retry==0:
            print(e)
        else:
            time.sleep(60*(6-retry))
            upload_file(file_path,parent_folder_id,retry-1)

class PathMamba(pl.LightningModule):
    
    def __init__(self, **kwargs):
        super().__init__()
        # self.save_hyperparameters(kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
        self.llm = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf").train()#.to("cuda:1")
        self.visual_encoder = vmamba_tiny_m2()#.to("cuda:0")
        self.tokenizer.pad_token_id = 0
        self.tokenizer.bos_token_id = 0
        self.embed_tokens = self.llm.get_input_embeddings()
        #self.projector = nn.Linear(self.visual_encoder.num_features, self.llm.config.hidden_size)#.to("cuda:0")
        self.norm = nn.LayerNorm(self.llm.config.hidden_size)#.to("cuda:0")
        self.tokenizer.padding_side = "right"
        self.val_step_outputs = []
        self.val_score = 0.0
        self.savedmodel_path = '/kaggle/working/'
        self.ckpt_file = None
        self.delta_file = None
        self.weights = [0.5, 0.5]
        self.scorer_types = ['Bleu_4', 'CIDEr']
        self.learning_rate = 1e-4
        self.gradient_clip_val = None
        self.beam_size = 3
        self.do_sample = False
        self.no_repeat_ngram_size = 2
        self.num_beam_groups = 1
        self.min_new_tokens = 80
        self.max_new_tokens = 120
        self.max_length = 100
        self.repetition_penalty = 2.0
        self.length_penalty = 2.0
        self.diversity_penalty = 0
        self.temperature = 0
        self.devices = 2
        self.num_nodes = 1
        self.accelerator = 'gpu'
        self.strategy = 'fsdp'
        self.precision = 'bf16-mixed'
        self.limit_val_batches = 1.0
        self.limit_test_batches = 1.0
        self.limit_train_batches = 1.0
        self.max_epochs = 3
        self.every_n_train_steps = 0
        self.val_check_interval = 1.0
        self.accumulate_grad_batches = 1
        self.num_sanity_val_steps = 2
        self.test = False
        self.validate = False
    
        for name, param in self.llm.named_parameters():
            param.requires_grad = False #Freezing llm params
        
    def score(self, ref, hypo):
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            #(Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores
    
    def encode_image(self, images):
        # image_embeds = []
        # for image in images:
        #     image_embed = self.visual_encoder(image)
        #     # print(image_embed.shape)
        #     image_embeds.append(image_embed)
        image_embeds = self.visual_encoder(images)
        # image_embeds = torch.stack(image_embeds).mean(0)image
        image_embeds = rearrange(image_embeds, 'b h w c -> b (h w) c')
        # print(image_embeds.shape)
        llm_inputs = image_embeds#self.projector(image_embeds)
        # print(llm_inputs.shape)
        # image_embed = self.norm(image_embed)
        llm_mask = torch.ones(llm_inputs.size()[:-1],dtype=torch.long).to(image_embeds.device)
        
        torch.cuda.empty_cache() 
        
        return llm_inputs, llm_mask
    
    def encode_prompt(self, image_embeds, questions, mask):
        batch_size = image_embeds.shape[0]
        prompt = f"Human: <Img><Image Here/><Img/> <Question Here/>\nAssistant:"
        question_embeds = self.tokenizer(questions, return_tensors = "pt", add_special_tokens= False, padding="max_length", truncation=True, max_length=158).to(image_embeds.device)
        pBefore, pAfter = prompt.split("<Image Here/>")
        pAfter0, pAfter1 = pAfter.split("<Question Here/>")
        pBefore = self.tokenizer(pBefore, return_tensors = "pt", add_special_tokens= False).to(image_embeds.device)
        pAfter0 = self.tokenizer(pAfter0, return_tensors = "pt", add_special_tokens= False).to(image_embeds.device)
        pAfter1 = self.tokenizer(pAfter1, return_tensors = "pt", add_special_tokens= False).to(image_embeds.device)
        pBefore = self.embed_tokens(pBefore.input_ids).to(torch.bfloat16).expand(batch_size, -1, -1) #feeling that we will not get 2d  
        pAfter0 = self.embed_tokens(pAfter0.input_ids).to(torch.bfloat16).expand(batch_size, -1, -1)
        pAfter1 = self.embed_tokens(pAfter1.input_ids).to(torch.bfloat16).expand(batch_size, -1, -1)
        question_embeds = self.embed_tokens(question_embeds.input_ids).to(torch.bfloat16)
        prompt = torch.cat([pBefore, image_embeds, pAfter0, question_embeds, pAfter1], dim=1)
        prompt_mask = mask[:,:1].expand(-1,prompt.shape[1])
        
        torch.cuda.empty_cache() 
        
        return prompt, prompt_mask
    
    def forward(self, batch):
        image_embeds, image_mask = self.encode_image(batch["image"])#.to(self.visual_encoder.device))
        image_embeds = self.norm(image_embeds)
        prompt_embeds, prompt_mask = self.encode_prompt(image_embeds, batch["question"], image_mask)

        answers = [answer+'</s>' for answer in batch["answer"]]
        answer_tokens = self.tokenizer(
            answers,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=155,
            add_special_tokens=False
        ).to(image_embeds.device)

        targets = answer_tokens.input_ids.masked_fill(answer_tokens.input_ids == 0, -100)
        empty_targets = torch.ones([prompt_mask.shape[0], prompt_mask.shape[1]+1], dtype=torch.long).to(image_embeds.device).fill_(-100)  # plus one for bos
        targets = torch.cat([empty_targets, targets], dim=1)#.to(self.llm.device)
        
        batch_size = image_embeds.shape[0]
        bos = torch.ones((batch_size,1),dtype=torch.long, device=image_embeds.device)*self.tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_mask = image_mask[:,:1]
        answer_embeds = self.embed_tokens(answer_tokens.input_ids)
        
        input_embeds = torch.cat([bos_embeds, prompt_embeds, answer_embeds], dim=1)#.to(self.llm.device)
        input_mask = torch.cat([bos_mask, prompt_mask, answer_tokens.attention_mask], dim=1)#.to(self.llm.device)
        
        torch.cuda.empty_cache() 
        
        outputs = self.llm(
            inputs_embeds = input_embeds.to(torch.bfloat16),
            attention_mask = input_mask,
            return_dict = True,
            labels = targets
        )
        
#         loss = outputs.loss
        return outputs

    def training_step(self, batch, batch_idx):
        result = self(batch)
        loss = {"loss":result.loss}
        self.log_dict(loss, prog_bar=True)
        return loss
    
    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        output_text = output_text.replace('!', '') # some model use "!" as special token, remove it
        return output_text
    
    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "state_dict": state_dict,
            # "config": self,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.savedmodel_path, 'checkpoints'), exist_ok=True)

        save_to = os.path.join(
            self.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'], eval_res['CIDEr']),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
        upload_file(save_to,parent_folder_id='1yvigs-SLFJQyrYJkURmkHqLFXoE3ASjo')
        
        
        
        
    def validation_step(self, batch, batch_idx):
        image_embeds, image_mask = self.encode_image(batch["image"])#.to(self.visual_encoder.device))
        image_embeds = self.norm(image_embeds)
        prompt_embeds, prompt_mask = self.encode_prompt(image_embeds, batch["question"], image_mask)
        
        answers = [answer+'</s>' for answer in batch["answer"]]
        answer_tokens = self.tokenizer(
            answers,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=155,
            add_special_tokens=False
        ).to(image_embeds.device)
        
        batch_size = image_embeds.shape[0]
        bos = torch.ones((batch_size,1),dtype=torch.long, device=image_embeds.device)*self.tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_mask = image_mask[:,:1]   
        input_embeds = torch.cat([bos_embeds, prompt_embeds], dim=1)#.to(self.llm.device)
        
        torch.cuda.empty_cache() 
        
        outputs = self.llm.generate(
            inputs_embeds = input_embeds.to(torch.bfloat16),
            num_beams=self.beam_size,
            do_sample=self.do_sample,
            min_new_tokens=self.min_new_tokens,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            temperature=self.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in answer_tokens.input_ids]
        # print(batch)
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": [i for i in range(32)]})
        return hypo, ref
        
    def on_validation_epoch_end(self):
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        result_folder = os.path.join(self.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{ global_step}" + '.json'), 'w', encoding='utf-8'),ensure_ascii=False)
        upload_file(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'),"1fE_AMukBmHn1PpHQ1fEdc-_3KrD4OHJK")
        
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w', encoding='utf-8'),ensure_ascii=False)
        # list_folder(parent_folder_id="1EuleXhyTwoaxY3DSmoSXxjJxpixbdSDR",delete=True)
        # upload_file(os.path.join(result_folder, 'refs.json'),"1EuleXhyTwoaxY3DSmoSXxjJxpixbdSDR")
        
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.scorer_types, self.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            self.save_checkpoint(eval_res)
            if val_score > self.val_score:
                self.val_score = val_score
        self.val_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    
    def get_progress_bar_dict(self):

        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()


def add_callbacks(args):
    log_dir = args.savedmodel_path
    os.makedirs(log_dir, exist_ok=True)
    
    # --------- Add Callbacks
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=os.path.join(log_dir, "checkpoints"),
    #     filename="{epoch}-{step}",
    #     save_top_k=-1,
    #     every_n_train_steps=args.every_n_train_steps,
    #     save_last=False,
    #     save_weights_only=False
    # )
    
    checkpoint_callback = CheckpointEveryNSteps(save_step_frequency=100)
    
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(log_dir, "logs"), name="tensorboard")
    csv_logger = CSVLogger(save_dir=os.path.join(log_dir, "logs"), name="csvlog")

    to_returns = {
        "callbacks": [ checkpoint_callback, lr_monitor_callback],
        "loggers": [csv_logger, tb_logger]
    }
    return to_returns


class DataModule(pl.LightningDataModule):
    
    def __init__(self):
        super().__init__()
#         self.args = args
        # self.img2tensor = Compose([Resize((256,256)),ToTensor()])  
        # self.drop_off_percent = drop_off_percent     
        self.dataset = load_dataset("flaviagiammarino/path-vqa")
        
        self.feature_extractor = AutoImageProcessor.from_pretrained('microsoft/swin-base-patch4-window7-224')
       

    # def prepare_data(self):
    #     self.dataset = load_dataset("flaviagiammarino/path-vqa")
        
    def transforms(self, batch):
        images = []
        for image in batch['image']:
            # with Image.open(os.path.join(self.args.base_dir, image_path)) as pil:
            array = np.array(image, dtype=np.uint8)
            if array.shape[-1] != 3 or len(array.shape) != 3:
                array = np.array(image.convert("RGB"), dtype=np.uint8)
            pixel_values = self.feature_extractor(array,return_tensors="pt",size=224).pixel_values
            image = pixel_values[0].to(torch.bfloat16)
            images.append(image)
        batch["image"] = images
             
        #batch['image'] = [self.img2tensor(img.convert('RGB')) for img in batch['image']]
        return batch
    
    def setup(self,stage):
        
        self.dataset.set_transform(self.transforms)  
        self.dataset['train'] = Subset(self.dataset['train'], list(range(600)))
        self.dataset['validation'] = Subset(self.dataset['validation'], list(range(400)))
        
    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=3, drop_last=True, pin_memory=True, num_workers=0)
    
    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=6, drop_last=False, pin_memory=False, num_workers=0)
    
    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=6, drop_last=False, pin_memory=True, num_workers=0)


class ArgsConfig:
    savedmodel_path = '/kaggle/working/'
    ckpt_file = '/kaggle/input/checkpth122/N-Step-Checkpoint_epoch3.ckpt' #"/kaggle/input/checkpth122/N-Step-Checkpoint_epoch0_global_step1300_batch_idx1296.ckpt"
    delta_file = None
    weights = [0.5, 0.5]
    scorer_types = ['Bleu_4', 'CIDEr']
    learning_rate = 1e-4
    gradient_clip_val = None
    beam_size = 3
    do_sample = False
    no_repeat_ngram_size = 2
    num_beam_groups = 1
    min_new_tokens = 80
    max_new_tokens = 120
    max_length = 100
    repetition_penalty = 2.0
    length_penalty = 2.0
    diversity_penalty = 0
    temperature = 0
    devices = 2
    num_nodes = 1
    accelerator = 'gpu'
    strategy = FSDPStrategy(state_dict_type="full", sharding_strategy = "FULL_SHARD")  #'ddp_find_unused_parameters_true'#'ddp'
    precision = 'bf16'
    limit_val_batches = 1.0
    limit_test_batches = 1.0
    limit_train_batches = 1.0
    max_epochs = 5
    every_n_train_steps = 0
    val_check_interval = 1.0
    accumulate_grad_batches = 1
    num_sanity_val_steps = 1
    test = False
    validate = False


def train(args):

    dm = DataModule()
    callbacks = add_callbacks(args)

    trainer = pl.Trainer(
        default_root_dir = "/kaggle/working/FSDP/",
        
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval = args.val_check_interval,
        limit_val_batches = args.limit_val_batches,
        max_epochs = args.max_epochs,
        num_sanity_val_steps = args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks["callbacks"], 
        logger=callbacks["loggers"]
    )

    

    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        if args.ckpt_file is not None:
            model = PathMamba(**args.__dict__)
            trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_file)
        else:
            model = PathMamba(**args.__dict__)
            trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    train(ArgsConfig())


