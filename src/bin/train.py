import os

import torch
try:
    from torch.cuda.amp import autocast
    autocast_available = True
except ImportError:
    class autocast:
        def __init__(self, enabled=True): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_value, exc_traceback): pass
    autocast_available = False

import transformers
from pathlib import Path
from src.modelling import ROOT
from src.modelling.optim import RAdam
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar
from torch.cuda.amp.grad_scaler import GradScaler
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from src.modelling.utils import instantiate_model_and_tokenizer, instantiate_loader
from src.modelling.evaluation import compute_span_srl_f1, write_props_format_predictions, predict_srl


def do_train(device, checkpoint=None, task="srl", task_type="span", duplicate_per_predicate=False, identify_predicate=False, fp16=False):

    model, tokenizer = instantiate_model_and_tokenizer(
        config['model'],
        checkpoint=checkpoint,
        dropout=config['dropout'],
        attention_dropout=config['attention_dropout'],
        from_pretrained=config['warm_start'],
        task=task,
        task_type=task_type,
    )

    # print(model)
    # print(model.config)

    if checkpoint is not None:
        print(f'Checkpoint restored ({checkpoint})!')

    optimizer = RAdam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'])
    if checkpoint is not None:
        optimizer.load_state_dict(torch.load(checkpoint)['optimizer'])

    if config['scheduler'] == 'cosine':
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=config['training_steps'])
    elif config['scheduler'] == 'constant':
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'])
    else:
        raise ValueError

    scaler = GradScaler(enabled=fp16)

    train_loader = instantiate_loader(
        config['train'],
        tokenizer,
        task=task,
        task_type=task_type,
        batch_size=config['batch_size'],
        evaluation=False,
        remove_longer_than=config['remove_longer_than'],
        duplicate_per_predicate=duplicate_per_predicate,
        identify_predicate=identify_predicate,
    )

    if not os.path.exists("data/tmp"):
        os.makedirs("data/tmp")
    if not os.path.exists("runs"):
        os.makedirs("runs")
    dev_gold_path = ROOT / 'data/tmp/dev-gold-{}-{}.txt'.format(task, task_type)
    dev_pred_path = ROOT / 'data/tmp/dev-pred-{}-{}.txt'.format(task, task_type)

    dev_loader = instantiate_loader(
        config['dev'],
        tokenizer,
        task=task,
        task_type=task_type,
        batch_size=config['batch_size'],
        evaluation=True,
        out=dev_gold_path,
        duplicate_per_predicate=duplicate_per_predicate,
        identify_predicate=identify_predicate
    )

    dev_loader, dev_gold_pred = dev_loader

    def train_step(engine, batch):
        model.train()
        x, y, extra = batch
        model.gr_mode = True
        with autocast(enabled=fp16):
            loss, *_ = model(**x, **y)
        scaler.scale((loss / config['accum_steps'])).backward()
        return loss.item()

    @torch.no_grad()
    def eval_step(engine, batch):
        model.eval()
        x, y, extra = batch
        model.gr_mode = True
        loss, *_ = model(**x, **y)
        return loss.item()

    trainer = Engine(train_step)
    evaluator = Engine(eval_step)

    @trainer.on(Events.STARTED)
    def update(engine):
        print('training started!')

    @trainer.on(Events.EPOCH_COMPLETED)
    @trainer.on(Events.ITERATION_COMPLETED(every=config['accum_steps']))
    def update(engine):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_trn_loss(engine):
        log_msg = f"training epoch: {engine.state.epoch}"
        log_msg += f" | loss_srl: {engine.state.metrics['trn_srl_loss']:.3f}"
        print(log_msg)

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_dev_eval(engine):
        dev_loader.batch_size = config['batch_size']
        dev_loader.device = next(model.parameters()).device
        evaluator.run(dev_loader)

    if not config['best_loss']:
        @evaluator.on(Events.EPOCH_COMPLETED)
        def metric_eval(engine): #F1
            device = next(model.parameters()).device
            dev_loader.device = device
            try:
                predictions = predict_srl(dev_loader, model, tokenizer, split=config['split'])
                write_props_format_predictions(dev_pred_path, predictions, dev_gold_pred)
                score = compute_span_srl_f1(dev_gold_path, dev_pred_path)
            except Exception as e:
                print(e, "This is why score is 0")
                score = 0.
            engine.state.metrics['dev_srl_score'] = score


    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_dev_loss(engine):
        log_msg = f"dev epoch: {trainer.state.epoch}"
        log_msg += f" | loss_srl: {engine.state.metrics['dev_srl_loss']:.3f}"
        if not config['best_loss']:
            log_msg += f" | F1_srl: {engine.state.metrics['dev_srl_score']:.3f}"
        print(log_msg)

    RunningAverage(output_transform=lambda out: out).attach(trainer, 'trn_srl_loss')
    RunningAverage(output_transform=lambda out: out).attach(evaluator, 'dev_srl_loss')

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names="all")
    if config['log_wandb']:
        from ignite.contrib.handlers.wandb_logger import WandBLogger
        wandb_logger = WandBLogger(init=False)
        wandb_logger.attach_output_handler(
            trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="iterations/trn_srl_loss",
            output_transform=lambda loss: loss
        )
        metric_names_trn = ['trn_srl_loss']
        metric_names_dev = ['dev_srl_loss']
        if not config['best_loss']:
            metric_names_dev.append('dev_srl_score')


        wandb_logger.attach_output_handler(
            trainer,
            event_name=Events.EPOCH_COMPLETED,
            tag="epochs",
            metric_names=metric_names_trn,
            global_step_transform=lambda *_: trainer.state.iteration,
        )

        wandb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="epochs",
            metric_names=metric_names_dev,
            global_step_transform=lambda *_: trainer.state.iteration,
        )

        @trainer.on(Events.ITERATION_COMPLETED)
        def wandb_log_lr(engine):
            wandb.log({'lr': scheduler.get_last_lr()[0]}, step=engine.state.iteration)

    if config['save_checkpoints']:

        if config['best_loss']:
            prefix = 'best-loss-srl'
            score_function = lambda x: 1 / evaluator.state.metrics['dev_srl_loss']
        else:
            prefix = 'best-srl-score'
            score_function = lambda x: evaluator.state.metrics['dev_srl_score']

        to_save = {'model': model, 'optimizer': optimizer}
        if config['log_wandb']:
            where_checkpoints = str(wandb_logger.run.dir)
        else:
            root = ROOT/'runs'/'runs'
            try:
                root.mkdir()
            except:
                pass
            where_checkpoints = root/str(len(list(root.iterdir())))
            try:
                where_checkpoints.mkdir()
            except:
                pass
            where_checkpoints = str(where_checkpoints)

        print(where_checkpoints)
        if config['best_loss']:
            n_saved = 1
        else:
            n_saved = 20
        handler = ModelCheckpoint(
            where_checkpoints,
            prefix,
            n_saved=n_saved,
            create_dir=True,
            score_function=score_function,
            global_step_transform=global_step_from_engine(trainer),
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler, to_save)

    model.to(device)
    device = next(model.parameters()).device
    train_loader.device = device
    trainer.run(train_loader, max_epochs=config['max_epochs'])

if __name__ == '__main__':

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import yaml
    import wandb

    parser = ArgumentParser(
        description="Trainer script",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--task-type',type=str, default="span", choices=['span', 'dep'], help='span-based vs dep-based')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/sweeped.yaml',
        help='Use the following config for hparams.')
    parser.add_argument('--checkpoint', type=str,
        help='Warm-start from a previous fine-tuned checkpoint.')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--duplicate-per-predicate', action='store_true', help="Activate flattened linearization.")
    parser.add_argument('--identify-predicate', action='store_true', help="Activate predicate identification in pipeline.")
    parser.add_argument('--device', type=str, default='cuda',  help="Device. 'cpu', 'cuda', 'cuda:<n>'.")
    args, unknown = parser.parse_known_args()

    device = torch.device(args.device)
    if args.identify_predicate:
        assert args.duplicate_per_predicate is False, "The option of identifying also the predicates is valid for the GSRL nested only. "
    if args.fp16 and autocast_available:
        raise ValueError('You\'ll need a newer PyTorch version to enable fp16 training.')

    with args.config.open() as y:
        config = yaml.load(y, Loader=yaml.FullLoader)

    if config['log_wandb']:
        wandb.init(
            entity=config["team"],
            project=config['wandb-project'],
            config=config,
            dir=str(ROOT / 'runs/'))
        config = wandb.config

    print(config)

    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        checkpoint = None

    do_train(
        device,
        checkpoint=checkpoint,
        task_type=args.task_type,
        duplicate_per_predicate=args.duplicate_per_predicate,
        identify_predicate=args.identify_predicate,
        fp16=args.fp16


    )
