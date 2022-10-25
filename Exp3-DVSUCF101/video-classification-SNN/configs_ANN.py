# configuration dics for SNN video classification
# the model is CNN-encoder(implemented by  LIAF) and LSTM-decoder
import numpy as np
import torch

training_configs = {
    'steps': 1,      # tensorboard steps
    'num_classes': 101,
    'total_epoch': 60,
    'batch_size': 64,
    'lr': 1e-4,
    'log_interval': 50,   # how many batch for print running loss and acc
    'save_interval': 30,   # how many epochs for saving checkpoints
    'begin_frame': 1,
    'end_frame': 16,
    'skip_frame': 1
}

selected_frames = np.arange(training_configs['begin_frame'], training_configs['end_frame'],
                            training_configs['skip_frame']).tolist()
time_windows = len(selected_frames)

model_configs = {
    'encoder': {
        'time_windows': 1,
        'actFun': torch.relu,
        'useBatchNorm': True,
        'input_size': [256, 342],
        'embed_dim': 512,
        # inChannels, outChannels, kernelSize, stride, padding, usePool, p_kernelSize, p_stride = self.cfgCnn[dice]
        'cfg_cnn': [(3, 8, (5, 5), (2, 2), (0, 0), False, 2, 2),
                    (8, 16, (3, 3), (2, 2), (0, 0), False, 2, 2),
                    (16, 32, (3, 3), (2, 2), (0, 0), False, 2, 2),
                    (32, 64, (3, 3), (2, 2), (0, 0), False, 2, 2),
                    ],
        'cfg_fc': [1024, 512],   # cfg_fc[-1] must be equal to embed_dim!
    },
    'decoder': {
        'num_layers': 1,
        'hidden_dim': 256,
        'num_classes': 101
    }
}


