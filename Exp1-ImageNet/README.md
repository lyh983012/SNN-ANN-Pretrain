# Rethinking the Pretraining as the Bridge from ANNs to SNNs

## Pipe-S

**Implementation of Pipe-S on ImageNet.**

1. To run the pretraining stage

    Set *traindir/valdir in train_pretrain.py* to the path of ImageNet train/val dataset. Then run

        python -m torch.distributed.launch --master_port=1234 --nproc_per_node=NUM-GPUs train_pretrain.py -net WSresnet18 -b 256 -lr 0.1

2. To run the finetune stage

    Set *Path in train_finetune.py* to the pretrained model reserved in the pretrain stage. Then, run

        python -m torch.distributed.launch --master_port=1234 --nproc_per_node=NUM-GPUs train_finetune.py -net Sresnet18 -b 256 -lr 0.1

The batchsize (-b) can be adjusted linearly to your GPU memory, and the learning rate should be adjusted accordingly.
