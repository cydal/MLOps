
## Pytorch Lightning 
### Without extra features


```
  | Name  | Type   | Params
---------------------------------
0 | model | ResNet | 11.2 M
---------------------------------
11.2 M    Trainable params
0         Non-trainable params
11.2 M    Total params
44.696    Total estimated model params size (MB)
Global seed set to 7
Epoch 0: 100%
197/197 [02:31<00:00, 1.30it/s, loss=1.44, v_num=8, val_loss=1.420, val_acc=0.469]
/usr/local/lib/python3.7/dist-packages/pytorch_lightning/core/datamodule.py:470: LightningDeprecationWarning: DataModule.teardown has already been called, so it will not be called again. In v1.6 this behavior will change to always call DataModule.teardown.
  f"DataModule.{name} has already been called, so it will not be called again. "
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Testing: 100%
40/40 [00:10<00:00, 3.72it/s]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'test_acc': 0.4690000116825104, 'test_loss': 1.4153896570205688}
--------------------------------------------------------------------------------
[{'test_acc': 0.4690000116825104, 'test_loss': 1.4153896570205688}]```


### DeviceStatsMonitor
##### Automatically monitors and logs device stats during training stage.

```python
from pytorch_lightning.callbacks import ModelSummary, DeviceStatsMonitor, EarlyStopping, ModelCheckpoint
```


```python
device_stats = DeviceStatsMonitor() 

# Setting the trainer specific arguments
trainer_args = {
    "logger": tboard,
    "checkpoint_callback": True,
    "callbacks": [lr_logger, early_stopping, 
                  checkpoint_callback, device_stats],
}

# Initiating the training process
trainer = Trainer(
    module_file="cifar10_train.py",
    data_module_file="cifar10_datamodule.py",
    module_file_args=args,
    data_module_args=data_module_args,
    trainer_args=trainer_args,
)
```



### Precision
##### mixed precision training for GPUs and CPUs for reduced memory footprint


```python
trainer = Trainer(
    max_epochs=1,
    gpus=AVAIL_GPUS,
    logger=TensorBoardLogger("lightning_logs/", name="extra_features"),
    callbacks=[LearningRateMonitor(logging_interval="step")],
    precision=16
)
```

```
  | Name  | Type   | Params
---------------------------------
0 | model | ResNet | 11.2 M
---------------------------------
11.2 M    Trainable params
0         Non-trainable params
11.2 M    Total params
22.348    Total estimated model params size (MB)
Global seed set to 7
Epoch 0: 100%
197/197 [02:59<00:00, 1.10it/s, loss=1.54, v_num=6, val_loss=1.550, val_acc=0.414]
/usr/local/lib/python3.7/dist-packages/pytorch_lightning/core/datamodule.py:470: LightningDeprecationWarning: DataModule.teardown has already been called, so it will not be called again. In v1.6 this behavior will change to always call DataModule.teardown.
  f"DataModule.{name} has already been called, so it will not be called again. "
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Testing: 100%
40/40 [00:11<00:00, 3.45it/s]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'test_acc': 0.4122999906539917, 'test_loss': 1.5447710752487183}
--------------------------------------------------------------------------------
[{'test_acc': 0.4122999906539917, 'test_loss': 1.5447710752487183}]
```





### RichProgressBar
##### Create a progress bar with rich text formatting.


```python
from pytorch_lightning.callbacks import RichProgressBar

# Initiating the training process
trainer = Trainer(
    max_epochs=1,
    gpus=AVAIL_GPUS,
    logger=TensorBoardLogger("lightning_logs/", name="extra_features"),
    callbacks=[LearningRateMonitor(logging_interval="step"), RichProgressBar()],
)
```

```

┏━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃   ┃ Name  ┃ Type   ┃ Params ┃
┡━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ 0 │ model │ ResNet │ 11.2 M │
└───┴───────┴────────┴────────┘
Trainable params: 11.2 M                                                                     
Non-trainable params: 0                                                                      
Total params: 11.2 M                                                                         
Total estimated model params size (MB): 44                                                   
Epoch 0    ━━━━━━━━━━━━━━━━━━━━━━━━━━ 197/197 0:02:31 • 0:00:00 2.09it/s loss: 1.44 v_num: 3 
Global seed set to 7
/usr/local/lib/python3.7/dist-packages/pytorch_lightning/core/datamodule.py:153: LightningDeprecationWarning: DataModule property `test_transforms` was deprecated in v1.5 and will be removed in v1.7.
  "DataModule property `test_transforms` was deprecated in v1.5 and will be removed in v1.7."
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Testing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 40/40 0:00:10 • 0:00:00 3.75it/s  
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'test_acc': 0.47350001335144043, 'test_loss': 1.4066499471664429}
--------------------------------------------------------------------------------
[{'test_acc': 0.47350001335144043, 'test_loss': 1.4066499471664429}]
```


### LR Finder
##### Auto LR Finder


```python

# Initiating the training process
trainer = Trainer(
    max_epochs=1,
    auto_lr_find=True,
    gpus=AVAIL_GPUS,
    logger=TensorBoardLogger("lightning_logs/", name="extra_features"),
    callbacks=[LearningRateMonitor(logging_interval="step")],
)
```



```
  | Name  | Type   | Params
---------------------------------
0 | model | ResNet | 11.2 M
---------------------------------
11.2 M    Trainable params
0         Non-trainable params
11.2 M    Total params
44.696    Total estimated model params size (MB)
Global seed set to 7
Epoch 0: 100%
197/197 [02:31<00:00, 1.30it/s, loss=1.39, v_num=7, val_loss=1.360, val_acc=0.497]
/usr/local/lib/python3.7/dist-packages/pytorch_lightning/core/datamodule.py:470: LightningDeprecationWarning: DataModule.teardown has already been called, so it will not be called again. In v1.6 this behavior will change to always call DataModule.teardown.
  f"DataModule.{name} has already been called, so it will not be called again. "
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Testing: 100%
40/40 [00:10<00:00, 3.72it/s]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'test_acc': 0.4927000105381012, 'test_loss': 1.3669812679290771}
--------------------------------------------------------------------------------
[{'test_acc': 0.4927000105381012, 'test_loss': 1.3669812679290771}]
```




### PRUNING
##### Technique to reduce model size and decrease inference requirements.


```python
from pytorch_lightning.callbacks import ModelPruning


# Initiating the training process
trainer = Trainer(
    max_epochs=1,
    auto_lr_find=True,
    gpus=AVAIL_GPUS,
    logger=TensorBoardLogger("lightning_logs/", name="extra_features"),
    callbacks=[LearningRateMonitor(logging_interval="step"), ModelPruning("l1_unstructured", amount=0.5)],
)
```


```
  | Name  | Type   | Params
---------------------------------
0 | model | ResNet | 11.2 M
---------------------------------
11.2 M    Trainable params
0         Non-trainable params
11.2 M    Total params
44.696    Total estimated model params size (MB)
Global seed set to 7
Epoch 0: 100%
197/197 [02:31<00:00, 1.30it/s, loss=1.37, v_num=11, val_loss=1.320, val_acc=0.502]
/usr/local/lib/python3.7/dist-packages/pytorch_lightning/core/datamodule.py:470: LightningDeprecationWarning: DataModule.teardown has already been called, so it will not be called again. In v1.6 this behavior will change to always call DataModule.teardown.
  f"DataModule.{name} has already been called, so it will not be called again. "
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Testing: 100%
40/40 [00:10<00:00, 3.71it/s]
--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'test_acc': 0.4812999963760376, 'test_loss': 1.3744076490402222}
--------------------------------------------------------------------------------
[{'test_acc': 0.4812999963760376, 'test_loss': 1.3744076490402222}]
```
