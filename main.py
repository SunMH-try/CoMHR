'''
 @FileName    : main.py
 @EditTime    : 2022-01-18 14:10:30
 @Author      : Buzhen Huang
 @Email       : hbz@seu.edu.cn
 @Description : 
'''
import vtkmodules
print(f"vtkmodules 版本:{vtkmodules.__version__}")
import os
import torch
from torch.utils.data import DataLoader
from cmd_parser import parse_config
from utils.module_utils import seed_worker, set_seed
from modules import init, LossLoader, ModelLoader, DatasetLoader
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
torch._dynamo.disable() 
# import builtins
# builtins.float = np.float_
# builtins.complex = np.complex_

# 自定义 collate_fn，过滤 None
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch) 

###########Load config file in debug mode#########
import sys
sys.argv = ['','--config=cfg_files/config.yaml'] #MoCap End2End HMAE_conv Lifting

def main(**args):
    seed = 7
    set_seed(seed)  # 固定 Python/numpy/torch 的 seed

    # 为 DataLoader 单独建 generator
    g = torch.Generator()
    g.manual_seed(seed)


    # Global setting
    dtype = torch.float32
    batchsize = args.get('batchsize')
    num_epoch = args.get('epoch')
    workers = args.get('worker')
    device = torch.device(index=args.get('gpu_index'), type='cuda')
    mode = args.get('mode')

    # Initialize project setting, e.g., create output folder, load SMPL model
    out_dir, logger, smpl = init(dtype=dtype, **args)

    # Load loss function
    loss = LossLoader(device=device, **args)

    # Load model
    model = ModelLoader(dtype=dtype, device=device, output=out_dir, **args)

    # create data loader
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
            collate_fn=collate_skip_none  # <- 加上这个
        )

        if args.get('use_sch'):
            model.load_scheduler(train_dataset.cumulative_sizes[-1])

    test_dataset = dataset.load_testset()
    test_loader = DataLoader(
        test_dataset,
        batch_size=batchsize, shuffle=False,
        num_workers=workers, pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_skip_none  # <- 加上这个
    )

    # Load handle function with the task name
    task = args.get('task')
    exec('from process import %s_train' %task)
    exec('from process import %s_test' %task)

    for epoch in range(num_epoch):
        # training mode
        if mode == 'train':
            training_loss = eval('%s_train' %task)(model, loss, train_loader, epoch, num_epoch, device=device)

            # # save trained model
            # if (epoch) % 1 == 0:
            #     model.save_model(epoch, task)

            if (epoch) % 1 == 0:
                testing_loss = eval('%s_test' %task)(model, loss, test_loader, device=device)
                lr = model.optimizer.state_dict()['param_groups'][0]['lr']
                logger.append([int(epoch + 1), lr, training_loss, testing_loss])
                # logger.plot(['Train Loss', 'Test Loss'])
                # savefig(os.path.join(out_dir, 'log.jpg'))
            else:
                testing_loss = -1.

            # save trained model
            if args.get('save_best', True):
                model.save_best_model(testing_loss, epoch, task)
            else:
                model.save_model(testing_loss, epoch, task)

        # testing mode
        elif epoch == 0 and mode == 'test':
            training_loss = -1.
            testing_loss = eval('%s_test' %task)(model, loss, test_loader, device=device)

            lr = model.optimizer.state_dict()['param_groups'][0]['lr']
            logger.append([int(epoch + 1), lr, training_loss, testing_loss])

    logger.close()


if __name__ == "__main__":
    args = parse_config()
    main(**args)





