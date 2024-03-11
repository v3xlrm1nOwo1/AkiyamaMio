import random
import time
import os

import numpy as np
import torch
import torch.optim as optim
import torch.cuda.amp as amp
from datasets import *

import utils
import config
import dataset
import mio_model
import prepare_dataset



if __name__ == '__main__':
    torch.manual_seed(config.RANDOM_SEED)
    torch.cuda.manual_seed_all(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # load and split dataset
    lyrics_dataset = load_dataset(config.DATASET_NAME, use_auth_token=config.ACCESS_TOKEN)
    lyrics_dataset = lyrics_dataset.shuffle(seed=config.RANDOM_SEED)

    train_testvalid = lyrics_dataset['train'].train_test_split(test_size=0.2)

    # split the 10% test + valid in half test, half valid
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)

    # gather everyone if you want to have a single DatasetDict
    lyrics_dataset = DatasetDict({
        'training': train_testvalid['train'],
        'test': test_valid['test'],
        'validation': test_valid['train']})

    # prepare dataset
    training_docs = prepare_dataset.get_docs(lyrics_dataset=lyrics_dataset['training'])
    validation_docs = prepare_dataset.get_docs(lyrics_dataset=lyrics_dataset['validation'])
    test_docs = prepare_dataset.get_docs(lyrics_dataset=lyrics_dataset['test'])

    # data loader
    training_dataloader = dataset.create_data_loader(stage='training', lyrics_tachi_tokens_ids=training_docs, batch_size=config.TRANING_BATCH_SIZE)
    validation_dataloader = dataset.create_data_loader(stage='validation', lyrics_tachi_tokens_ids=validation_docs, batch_size=config.VALIDATION_BATCH_SIZE)
    test_dataloader = dataset.create_data_loader(stage='test', lyrics_tachi_tokens_ids=test_docs, batch_size=config.TEST_BATCH_SIZE)

    # model 
    model = mio_model.model
    model = model.to(config.DEVICE)

    # build optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'ln']  # no decay for bias and LayerNorm (ln)
    
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': config.L2_PENALTY},
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=config.L2_PENALTY
    )

    # build lr scheduler
    lr_scheduler = utils.get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.NUM_WARMUP_STEPS,
        num_training_steps=config.NUM_TRAINING_STEPS,
    )

    n_step = 0
    start_n_epoch = 0
    best_ppl = float('inf')

    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    start_time = time.time()
    trn_reporter = utils.StatisticsReporter()
    eval_reporter = utils.StatisticsReporter()

    if config.USE_AMP:
        scaler = amp.GradScaler()

    torch.cuda.empty_cache()

    for epoch_idx in range(start_n_epoch, config.NUM_EPOCHS):
        for batch_data in training_dataloader:
            n_step += 1

            # stop if reaches the maximum tranining step
            if n_step >= config.NUM_TRAINING_STEPS:
                break

            # forward
            model.train()
            with amp.autocast():
                loss, ppl = mio_model.forward_step(model=model, tokenizer=config.TOKENIZER, batch_data=batch_data)

            trn_reporter.update_data({'ppl': ppl.item(), 'loss': loss.item()})

            # backward
            loss /= config.NUM_ACCUM_STEPS

            if config.USE_AMP:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            del loss

            if n_step % config.NUM_ACCUM_STEPS == 0:
                # clip gradient
                if config.MAX_GRAD_NORM > 0.0:
                    if config.USE_AMP:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

                # update model parameters
                if config.USE_AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                # zero gradients
                optimizer.zero_grad()

            # check loss
            if n_step > 0 and n_step % config.CHECK_LOSS_AFTER_NUM_STEP == 0:
                lr = list(lr_scheduler.optimizer.param_groups)[0]['lr']
                log = f'{time.time() - start_time:.2f}s Epoch {epoch_idx}, step {n_step}, lr {lr:.5g} - '
                log += trn_reporter.to_string()
                utils.log_data(log)
                trn_reporter.clear()

            # save   
            if n_step > 0 and n_step % config.SAVE_AFTER_NUM_STEP == 0:
                model.eval()
                model_to_save = model.module if hasattr(model, 'module') else model

                with torch.no_grad():
                    for eval_batch_idx, eval_batch_data in enumerate(validation_dataloader):
                        with amp.autocast():
                            loss, ppl = mio_model.forward_step(model=model_to_save, tokenizer=config.TOKENIZER, batch_data=eval_batch_data)
                        eval_reporter.update_data({'ppl': ppl.item(), 'loss': loss.item()})

                        if eval_batch_idx == len(validation_dataloader) - 1:
                            break
                del loss

                log = f'<Validation> - {time.time() - start_time:.3f}s - '
                log += eval_reporter.to_string()
                utils.log_data(log)

                random_sample = random.choice(lyrics_dataset['validation'])
                lyrics_gen = utils.gen_anime_lyrics(title=random_sample['SongTitle'], prompt_text=random_sample['Start Singing'], model=model)
                print(lyrics_gen)

                # save current model
                checkpoint = {
                    'model': model_to_save.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'n_epoch': epoch_idx,
                    'n_step': n_step,
                    'best_ppl': best_ppl
                }

                torch.save(
                    checkpoint,
                    f'{config.OUTPUT_DIR}/last.checkpoint'
                )

                utils.log_data(f'checkpoint saved to {config.OUTPUT_DIR}/last.checkpoint')

                # save best model
                cur_ppl = eval_reporter.get_value('ppl')
                if cur_ppl < best_ppl:
                    best_ppl = cur_ppl

                    torch.save(
                        checkpoint,
                        f'{config.OUTPUT_DIR}/best.checkpoint'
                    )
                    utils.log_data(f'best checkpoint saved to {config.OUTPUT_DIR}/best.checkpoint')
                eval_reporter.clear()

            # decay learning rate
            lr_scheduler.step()