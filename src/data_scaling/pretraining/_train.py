from model._transformer import FoundationalModel
from utils._dataset import TransformerDataset, ParquetDataset
from dataloader.datamodules import MerlinDataModule, MerlinDataModuleDistributed
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.distributed import get_global_rank, get_world_size, get_rank
import wandb
import torch
    
def manual_train_fm(config=None):
    
    pl.seed_everything(42)
    
    model = FoundationalModel(dim_model=config['dim_model'], 
                              nheads=config['nheads'], 
                              dim_feedforward=config['dim_feedforward'], 
                              nlayers=config['nlayers'],
                              dropout=config['dropout'],
                              batch_first=config['batch_first'], 
                              masking_p=config['masking_p'], 
                              n_tokens=config['n_tokens'],
                              context_length=config['context_length'],
                              warmup=config['warmup'],
                              lr=config['lr'],
                              batch_size=config['batch_size'],
                              max_epochs=config['max_epochs'],
                              autoregressive=config['autoregressive'],
                              pool=config['pool'],
                              supervised_task=config['supervised_task'],
                              karpathy=config['karpathy'],
                              specie=config['specie'],
                              assay=config['assay'],
                              modality=config['modality'],
                              contrastive=config['contrastive'])
        
    if config['pretrained_path'] is not None:
        print("Loading pretrained model")
        model = FoundationalModel.load_from_checkpoint(checkpoint_path=config['pretrained_path'],
                                                    #    dim_model=config['dim_model'], 
                                                    #     nheads=config['nheads'], 
                                                    #     dim_feedforward=config['dim_feedforward'], 
                                                    #     nlayers=config['nlayers'],
                                                    #     dropout=config['dropout'],
                                                    #     batch_first=config['batch_first'], 
                                                    #     masking_p=config['masking_p'], 
                                                    #     n_tokens=config['n_tokens'],
                                                    #     context_length=config['context_length'],
                                                        warmup=config['warmup'],
                                                        lr=config['lr'],
                                                    #     max_epochs=config['max_epochs'],
                                                    #     autoregressive=config['autoregressive'],
                                                    #     pool=config['pool'],
                                                    #     supervised_task=config['supervised_task'],
                                                    #     karpathy=config['karpathy'])
        )
    
    wandb_logger = WandbLogger(project=f'FM-{config["organ"]}', entity='nicheformer')
    
    checkpoint_callback = ModelCheckpoint(dirpath=f'/lustre/groups/ml01/projects/2023_nicheformer/pretrained_models/{config["organ"]}_heads_{config["nheads"]}_blocks_{config["nlayers"]}_maxsteps_{config["max_epochs"]}/', every_n_train_steps=10000, monitor='train_loss', save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
                        logger=wandb_logger,
                        accelerator='gpu',
                        max_epochs=1000,
                        devices=-1,
                        #num_nodes=3,
                        log_every_n_steps=100,
                        check_val_every_n_epoch=50,
                        strategy="ddp_find_unused_parameters_true",
                        default_root_dir=f'/home/icb/alejandro.tejada/spatial-transformer/trained_model_heads_{config["nheads"]}_blocks_{config["nlayers"]}/',
                        callbacks=[checkpoint_callback, lr_monitor],
                        precision='bf16-mixed',
                        gradient_clip_val=0.5,
                        accumulate_grad_batches=10)
    
    #path_organ = '/lustre/groups/ml01/projects/spatial_transformer/merlin_cxg_2023_05_15_tokenized/'
    #key_organ = 'cell_type'
    
    path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/cellxgene_census_tokenized'
    key_organ = 'cell_type'
    splits=True
     
    if config['organ'] == 'everything':
        path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/data/nicheformer_tokens_capped'
        key_organ = ['X', 'specie', 'assay', 'modality']
        splits = True
        
    # path_organ = '/lustre/groups/ml01/projects/2023_nicheformer/data/test'
    # key_organ = ['X', 'specie', 'assay', 'modality']
    # splits=True
    
    print(f"Using path {path_organ}")
    
    module = MerlinDataModuleDistributed(path=path_organ, 
                        columns=key_organ,#['X', 'specie', 'assay', 'modality'],
                        batch_size=config['batch_size'],
                        world_size=trainer.world_size,
                        splits=splits,
                        skip_steps=config['skip_steps'])
    
    if config['pretrained_path'] is not None and config['retake_training']:
        print(f"Training model in {config['organ']} from checkpoint!")
        
        # Add option to skip initial batches when resuming training
        if 'skip_steps' in config and config['skip_steps'] > 0:
            print(f"Skipping first {config['skip_steps']} batches of training")
            # Create a custom callback to skip batches
            class SkipBatchesCallback(pl.Callback):
                def __init__(self, num_batches_to_skip):
                    super().__init__()
                    self.num_batches_to_skip = num_batches_to_skip
                    self.batches_skipped = 0
                
                def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                    if self.batches_skipped < self.num_batches_to_skip:
                        self.batches_skipped += 1
                        return -1  # Skip this batch
                    return None
            
            # Add the callback to the trainer
            skip_callback = SkipBatchesCallback(config['skip_steps'])
            trainer.callbacks.append(skip_callback)
            
        trainer.fit(model=model, datamodule=module, ckpt_path=config['pretrained_path'])
        
    print(f"World size: {trainer.world_size}")
    print(f"Training model in {config['organ']} from scratch")

    #model = torch.compile(model, mode='max-autotune')
    trainer.fit(model=model, datamodule=module)
