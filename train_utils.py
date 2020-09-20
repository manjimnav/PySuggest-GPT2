import os
import time

import more_itertools as mit
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

from apex import amp
from metrics_utils import topk_acc, accuracy_score


def get_optimizer(args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.epsilon)

    return optimizer


def get_inputs(args, batch, device):
    inputs = batch[:, 0, :]

    inputs = inputs.to(device)
    result = {}

    if 'xlnet' in args.model_name:
        # inputs.append(args.vocab_size + 1)
        perm_mask = torch.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[1]), dtype=torch.float).to(device)
        perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
        target_mapping = torch.zeros((inputs.shape[0], 1, inputs.shape[1]), dtype=torch.float)
        target_mapping[:, 0, -1] = 1.0

        result['perm_mask'] = perm_mask.to(device)
        result['target_mapping'] = target_mapping.to(device)
        result['labels'] = batch[:, 1, -1].unsqueeze(0).to(device)
    elif 'bert' in args.model_name:
        result['masked_lm_labels'] = batch[:, 0, :].to(device)
        result['labels'] = batch[:, 1, :].to(device)
    else:
        result['labels'] = batch[:, 1, :].to(device)

    result['input_ids'] = inputs

    return result


def evaluate(args, dataloader, model, device):
    model.eval()
    acc = 0
    top5_acc = 0
    eval_loss = 0
    nb_eval_steps = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        kargs = get_inputs(args, batch, device)
        labels = kargs['labels']
        if 'bert' in args.model_name:
            print('deleted')
            del kargs['labels']
        with torch.no_grad():

            if any(mn in args.model_name for mn in args.transformers_models):
                outputs = model(**kargs)
                # print("Predict time: {}".format((time.time() - start_time)/60))
                lm_loss = outputs[0]  # Crossentropy loss
                logits = outputs[1]  # Vocabulary probabilities
            elif 'seq2seq' in args.model_name:
                outputs = model(kargs['input_ids'], labels)
                # print("Predict time: {}".format((time.time() - start_time)/60))
                lm_loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))  # Crossentropy loss
                logits = outputs  # Vocabulary probabilities
                #print(logits.size())
            else:
                try:
                    outputs = model(kargs['input_ids'])
                except Exception as e:
                    print(e)
                    continue
                # outputs = outputs.argmax(2)
                # print("Predict time: {}".format((time.time() - start_time)/60))
                lm_loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))  # Crossentropy loss
                logits = outputs  # Vocabulary probabilities
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

        acc += accuracy_score(logits, labels)
        top5_acc += topk_acc(logits, labels)
        #print(acc/nb_eval_steps)
        #print(top5_acc/nb_eval_steps)

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss)) / nb_eval_steps
    accuracy_value = acc / nb_eval_steps
    top5_acc = top5_acc / nb_eval_steps

    return perplexity.item(), accuracy_value.item(), top5_acc.item(), eval_loss


def generate_results(*args):
    results = pd.DataFrame(data=[])

    results['Exec time'] = args[0]
    results['Train Loss'] = args[1]
    results['Train Acc'] = args[2]
    results['Train Top 5 Acc'] = args[3]
    results['Train PP'] = args[4]

    if len(args) > 6:
        results['Eval Loss'] = args[5]
        results['Eval Acc'] = args[6]
        results['Eval Top 5 Acc'] = args[7]
        results['Eval PP'] = args[8]

    results['Epoch'] = args[-1]

    results.set_index('Epoch')

    return results


def train(train_dataloader, model, device, args, eval_dataloader=None):
    vocab_size = args.vocab_size
    if any(mn in args.model_name for mn in args.transformers_models if mn not in 'transfo-xl-wt103'):
        model.resize_token_embeddings(vocab_size)

    optimizer = get_optimizer(args, model)

    # Work with 16bit precission
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
    # Params initialization
    exec_time = 0
    epochs_trained = 0
    gradient_accumulation_steps = args.gradient_accumulation_steps
    global_step = 0
    if args.train_len is not None:
        train_len = args.train_len
    else:
        train_len = mit.ilen(x for x in train_dataloader)  # Calculate the size of the data iterator
    print("Train length: " + str(train_len))

    t_total = train_len // gradient_accumulation_steps * args.epochs
    exec_times = []
    train_accuracies, train_losses, train_top5_accuracies, train_pp = [], [], [], []
    eval_accuracies, eval_losses, eval_top5_accuracies, eval_pps = [], [], [], []
    steps_trained_in_current_epoch = 0
    # lr Scheduler initialization
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    checkpoint_prefix = "checkpoint"

    # Restore training
    if (
            args.output_dir
            and os.path.exists(args.output_dir)
            and os.path.isfile(os.path.join(args.output_dir, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.output_dir, "scheduler.pt"))
            and args.restore_training
    ):
        last_checkpoint = os.listdir(args.output_dir)[-1]
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.output_dir, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.output_dir, "scheduler.pt")))

        # set global_step to gobal_step of last saved checkpoint from model path
        checkpoint_suffix = last_checkpoint.split("-")[-1].split("/")[0]
        global_step = int(checkpoint_suffix)
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

    print("***** Running training *****")
    train_iterator = trange(
        epochs_trained, args.epochs, desc="Epoch"
    )
    # Main loop
    for epoch in train_iterator:

        start_time = time.time()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        iterations = 0
        acc = 0
        top5_acc = 0
        crossentropy = 0

        for step, batch in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            # print("Batch time: {}".format((time.time() - start_time)/60))
            model.train()
           
            kargs = get_inputs(args, batch, device)
            
            labels = kargs['labels']
            if 'bert' in args.model_name:
                del kargs['labels']
            # print("Get imput time: {}".format((time.time() - start_time)/60))
            if any(mn in args.model_name for mn in args.transformers_models):
                outputs = model(**kargs)
                # print("Predict time: {}".format((time.time() - start_time)/60))
                loss = outputs[0]  # Crossentropy loss
                logits = outputs[1]  # Vocabulary probabilities
            elif 'seq2seq' in args.model_name:
                outputs = model(kargs['input_ids'], labels)
                # print("Predict time: {}".format((time.time() - start_time)/60))
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))  # Crossentropy loss
                logits = outputs  # Vocabulary probabilities
                #print(logits.size())
            else:
                try:
                    outputs = model(kargs['input_ids'])
                except Exception as e:
                    print(e)
                    continue
                # print("Predict time: {}".format((time.time() - start_time)/60))
                loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))  # Crossentropy loss
                logits = outputs  # Vocabulary probabilities

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            crossentropy += loss.item()
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                if args.fp16:
                    clip_grad_norm_(amp.master_params(optimizer), 1.0)
                else:
                    clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                iterations += 1
            if global_step % args.train_logs_freq == 0 and args.show_logs_during_train:
                # Metric calculation
                crossentropy_temp = crossentropy / iterations
                accuracy_value_temp = acc / iterations
                top5_acc_temp = top5_acc / iterations
                pp_temp = torch.exp(torch.tensor(crossentropy_temp))

                print(  # 'Epoch: {}/{}'.format(epoch, epochs),
                    'Time: {}'.format((time.time() - start_time) / 60),
                    'Train Loss: {}'.format(crossentropy_temp),
                    'Train Acc: {}'.format(accuracy_value_temp),
                    'Train Acc@5: {}'.format(top5_acc_temp),
                    'Train PP: {}'.format(pp_temp)
                )

            # Saving model
            if args.save_steps > 0 and global_step % args.save_steps == 0 and args.checkpoint_model:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                os.makedirs(output_dir, exist_ok=True)
                if any(mn in args.model_name for mn in args.transformers_models):
                    model.save_pretrained(output_dir)
                else:
                    torch.save(model.state_dict(), output_dir)
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                print("Saving optimizer and scheduler states to %s", output_dir)

            # Update metrics
            acc += accuracy_score(logits, labels)
            top5_acc += topk_acc(logits, labels)

        eval_string = ''
        if eval_dataloader is not None:
            eval_pp, eval_acc, eval_top5_acc, eval_loss = evaluate(args, eval_dataloader, model, device)
            eval_string = ('|', 'Eval Loss: {}'.format(eval_loss),
                           'Eval Acc: {}'.format(eval_acc),
                           'Eval Acc@5: {}'.format(eval_top5_acc),
                           'Eval PP: {}'.format(eval_pp))

        # Metric calculation
        crossentropy = crossentropy / iterations
        accuracy_value = acc / iterations
        top5_acc = top5_acc / iterations
        exec_time += (time.time() - start_time) / 60
        pp = torch.exp(torch.tensor(crossentropy))

        print(  # 'Epoch: {}/{}'.format(epoch, epochs),
            'Time: {}'.format(exec_time),
            'Train Loss: {}'.format(crossentropy),
            'Train Acc: {}'.format(accuracy_value),
            'Train Acc@5: {}'.format(top5_acc),
            'Train PP: {}'.format(pp),
            *eval_string
        )

        exec_times.append(exec_time)
        train_losses.append(crossentropy)
        train_accuracies.append(accuracy_value.item())
        train_top5_accuracies.append(top5_acc.item())
        train_pp.append(pp.item())
        if eval_dataloader is not None:
            eval_losses.append(eval_loss)
            eval_accuracies.append(eval_acc)
            eval_top5_accuracies.append(eval_top5_acc)
            eval_pps.append(eval_pp)

    results = generate_results(exec_times, train_losses, train_accuracies,
                               train_top5_accuracies, train_pp, eval_losses, eval_accuracies,
                               eval_top5_accuracies, eval_pps, list(range(1, args.epochs + 1)))

    return model, results
