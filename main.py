import vtkmodules
import os
import torch
import sys
import numpy as np
from torch.utils.data import DataLoader
from cmd_parser import parse_config
from utils.module_utils import seed_worker, set_seed
from modules import init, LossLoader, ModelLoader, DatasetLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch._dynamo.disable()

# Custom collate function to filter None samples
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# Debug mode configuration
sys.argv = ['', '--config=cfg_files/config.yaml']

def main(**args):
    seed = 7
    set_seed(seed)

    # Generator for DataLoader reproducibility
    g = torch.Generator()
    g.manual_seed(seed)

    # Global settings
    dtype = torch.float32
    batchsize = args.get('batchsize')
    num_epoch = args.get('epoch')
    workers = args.get('worker')
    device = torch.device(index=args.get('gpu_index'), type='cuda')
    mode = args.get('mode')

    # Initialize project settings and SMPL model
    out_dir, logger, smpl = init(dtype=dtype, **args)

    # Load loss and model
    loss = LossLoader(device=device, **args)
    model = ModelLoader(dtype=dtype, device=device, output=out_dir, **args)

    # Data loader setup
    dataset = DatasetLoader(dtype=dtype, smpl=smpl, **args)
    if mode == 'train':
        train_dataset = dataset.load_trainset()
        train_loader = DataLoader(
            train_dataset,
            batch_size=batchsize,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
            collate_fn=collate_skip_none
        )

        if args.get('use_sch'):
            model.load_scheduler(train_dataset.cumulative_sizes[-1])

    test_dataset = dataset.load_testset()
    test_loader = DataLoader(
        test_dataset,
        batch_size=batchsize, 
        shuffle=False,
        num_workers=workers, 
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_skip_none,
    )

    # Dynamic loading of task functions
    task = args.get('task')
    exec('from process import %s_train' %task)
    exec('from process import %s_test' %task)

    for epoch in range(num_epoch):
        if mode == 'train':
            # Training phase
            training_loss = eval('%s_train' %task)(model, loss, train_loader, epoch, num_epoch, device=device)

            # Testing phase during training
            if (epoch) % 1 == 0:
                testing_loss = eval('%s_test' %task)(model, loss, test_loader, device=device)
                lr = model.optimizer.state_dict()['param_groups'][0]['lr']
                logger.append([int(epoch + 1), lr, training_loss, testing_loss])
            else:
                testing_loss = -1.

            # Model saving logic
            if args.get('save_best', True):
                model.save_best_model(testing_loss, epoch, task)
            else:
                model.save_model(testing_loss, epoch, task)

        elif mode == 'test':
            # Pure testing mode
            training_loss = -1.
            result = eval('%s_test' % task)(model, loss, test_loader, device=device)

            # Handle different return types safely
            if isinstance(result, tuple):
                testing_loss, folder_loss = result
            else:
                testing_loss, folder_loss = result, {}

            if torch.is_tensor(testing_loss):
                testing_loss = testing_loss.item()

            lr = model.optimizer.state_dict()['param_groups'][0]['lr']
            logger.append([int(epoch + 1), lr, training_loss, testing_loss])

    logger.close()

if __name__ == "__main__":
    args = parse_config()
    main(**args)