import torch
from pathlib import Path
from src.modelling import ROOT
from src.modelling.evaluation import predict_srl, compute_span_srl_f1, write_props_format_predictions, \
    overwrite_gold_with_predicted, compute_dep_srl_f1
from src.modelling.utils import instantiate_loader, instantiate_model_and_tokenizer


if __name__ == '__main__':

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(description="Script to predict SRL spans given sentences. CONLL format as input.", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datasets', type=str, required=True,
        help="Required. One or more glob patterns to use to load srl files.")
    parser.add_argument('--checkpoint', type=str, required=True,
        help="Required. Checkpoint to restore.")
    parser.add_argument('--model', type=str, default='facebook/bart-large',
        help="Model config to use to load the model class.")
    parser.add_argument('--beam-size', type=int, default=1,
        help="Beam size.")
    parser.add_argument('--batch-size', type=int, default=1000,
        help="Batch size (as number of linearized  tokens per batch).")
    parser.add_argument('--device', type=str, default='cuda',
        help="Device. 'cpu', 'cuda', 'cuda:<n>'.")
    parser.add_argument('--pred-path', type=str, default=ROOT / 'data/tmp/test-pred-srl-only.txt',
        help="Where to write predictions.")
    parser.add_argument('--gold-path', type=str, default=ROOT / 'data/tmp/test-gold-srl-only.txt',
        help="Where to write the gold file.")
    parser.add_argument('--return-all', action='store_true')
    parser.add_argument('--nosplit', action="store_true")
    parser.add_argument('--duplicate-per-predicate', action='store_true', help="Activate flattened linearization.")
    parser.add_argument('--identify-predicate', action='store_true',
                        help="Activate predicate identification in pipeline.")
    parser.add_argument('--task-type',type=str, default="span", choices=['span', 'dep'], help='span-based vs dep-based')
    parser.add_argument('--eval-name',type=str, default="span")


    args = parser.parse_args()

    if args.identify_predicate:
        assert args.duplicate_per_predicate is False, "The option of identifying also the predicates is valid for the GSRL nested only. "

    device = torch.device(args.device)
    model, tokenizer = instantiate_model_and_tokenizer(
        args.model,
        dropout=0.,
        attention_dropout=0.,
        task="srl",
        task_type=args.task_type
    )
    model.gr_mode = True
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'])
    model.to(device)

    gold_path = args.gold_path
    pred_path = args.pred_path
    loader, gold_pred = instantiate_loader(
        args.datasets,
        tokenizer,
        task="srl",
        batch_size=args.batch_size,
        evaluation=True,
        out=gold_path,
        duplicate_per_predicate=args.duplicate_per_predicate,
        task_type=args.task_type,
        identify_predicate=args.identify_predicate
    )
    loader.device = device
    predictions = predict_srl(
        loader,
        model,
        tokenizer,
        beam_size=args.beam_size,
        return_all=args.return_all,
        split=False if args.nosplit else True,
        task_type=args.task_type
    )
    if args.return_all:
        predictions = [g for gg in predictions for g in gg]
    if args.task_type == "span":
        write_props_format_predictions(pred_path, predictions, gold_pred)
        if not args.return_all:
            score = compute_span_srl_f1(gold_path, pred_path)
            print(f'F1: {score:.3f}')
    else:
        overwrite_gold_with_predicted(pred_path, predictions, gold_path)
        if not args.return_all:
            score = compute_dep_srl_f1(gold_path, pred_path, args.eval_name)
            print(f'F1: {score:.4f}')

