import os
import sys
import time
import random
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
#from utils.losses import DiceLoss
from monai.losses import DiceLoss
# Knowledge Distillation Loss Functions



import torch.nn.functional as F
'''
def distillation_loss(student_outputs, teacher_outputs, labels, alpha=0.5):
    mse_loss = F.mse_loss(F.softmax(student_outputs, dim=1), F.softmax(teacher_outputs, dim=1))
    ce_loss = F.cross_entropy(student_outputs, labels)
    return (1 - alpha) * ce_loss + alpha * mse_loss

'''
def distillation_loss(student_outputs, teacher_outputs, labels, temperature=3.0, alpha=0.5):
    """
    Compute the distillation loss using KL divergence with numerical stability.
    """
    # Normalize outputs to prevent instability
    student_outputs = student_outputs / torch.norm(student_outputs, dim=1, keepdim=True)
    teacher_outputs = teacher_outputs / torch.norm(teacher_outputs, dim=1, keepdim=True)

    # Soften the outputs using temperature
    student_probs = F.log_softmax(student_outputs / temperature, dim=1)
    teacher_probs = F.softmax(teacher_outputs / temperature, dim=1)

    # Compute KL divergence loss (distillation loss)
    kld_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

    # Compute cross-entropy loss (hard labels)
    ce_loss = F.cross_entropy(student_outputs, labels)

    # Combine the losses
    total_loss = (1 - alpha) * ce_loss + alpha * kld_loss
    return total_loss


def compute_kd_response_loss(student_outputs, teacher_outputs, temperature=3.0):
    """
    Computes KL divergence loss for response distillation (soft labels from teacher).
    """
    student_probs = F.log_softmax(student_outputs / temperature, dim=1)
    teacher_probs = F.softmax(teacher_outputs / temperature, dim=1)
    return F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)


def compute_kd_attention_map_loss(batch_size, student_model, teacher_model, layers):
    """
    Computes attention map loss between student and teacher models.
    """
    attn_loss = 0.0
    for layer in layers:
        student_attn = student_model.get_attention_map(layer)  # Implement this in model
        teacher_attn = teacher_model.get_attention_map(layer)  # Implement this in model
        attn_loss += F.mse_loss(student_attn, teacher_attn)
    return attn_loss / len(layers)


def trainer_synapse(args, teacher_model, student_model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # Training dataset and DataLoader
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        teacher_model = nn.DataParallel(teacher_model)
        student_model = nn.DataParallel(student_model)

    teacher_model.eval()  # Teacher model is in evaluation mode
    student_model.train()  # Student model is in training mode

    #ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(student_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # Forward pass through teacher and student models
            with torch.no_grad():
                teacher_outputs = teacher_model(image_batch)
            student_outputs = student_model(image_batch)

            # Compute distillation loss
            loss = distillation_loss(student_outputs, teacher_outputs, label_batch[:].long())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)

            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(student_outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(student_model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(student_model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"