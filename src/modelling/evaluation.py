import os
import copy
import torch
import datetime
import subprocess
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
from src.modelling.utils import load_nlp_pipeline


def predict_srl(loader, model, tokenizer,  beam_size=1, task_type="span", tokens=None, return_all=False, split=False):
    if split:
        nlp = load_nlp_pipeline()
    else:
        nlp = None
    shuffle_orig = loader.shuffle
    sort_orig = loader.sort

    loader.shuffle = False
    loader.sort = True

    total = len(loader.dataset)
    model.eval()
    model.gr_mode = True
    if tokens is None:
        ids = []
        tokens = []
        input_sentences = []
        gold_linearizations = []
        with tqdm(total=total) as bar:
            for x, y, extra in loader:
                ii = extra['ids']
                input_sent = [inst['sentence'] for inst in extra['structure']]
                gold_lin = [inst for inst in extra['linearized_structure']]
                ids.extend(ii)
                input_sentences.extend(input_sent)
                gold_linearizations.extend(gold_lin)
                nseq = len(ii)
                with torch.no_grad():
                    out = model.generate(
                        **x,
                        max_length=1024,
                        decoder_start_token_id=0,
                        num_beams=beam_size,
                        num_return_sequences=beam_size)
                    for i1 in range(0, out.size(0), beam_size):
                        tokens_same_source = []
                        tokens.append(tokens_same_source)
                        for i2 in range(i1, i1+beam_size):
                            tokk = out[i2].tolist()
                            tokens_same_source.append(tokk)
                bar.update(nseq)
    # reorder
    tokens = [tokens[i] for i in ids]
    tokens = [t for tt in tokens for t in tt]

    predictions = []
    index_idx = 0
    for i1 in range(0, len(tokens), beam_size):
        structure_same_source = []
        predictions.append(structure_same_source)
        for i2 in range(i1, i1+beam_size):
            tokk = tokens[i2]
            status, instance, lin = tokenizer.decode_srl(tokk, input_sentences[index_idx], gold_linearizations[index_idx], nlp, task_type=task_type)
            instance["status"] = status
            instance["pred_lin"] = lin
            instance["gold_lin"] = gold_linearizations[index_idx]
            instance["input_sentence"] = input_sentences[index_idx]

            structure_same_source.append(instance)
        structure_same_source[:] = tuple(zip(*sorted(enumerate(structure_same_source), key=lambda x: (x[1]['status'].value, x[0]))))[1]
        index_idx+=1
    loader.shuffle = shuffle_orig
    loader.sort = sort_orig

    if not return_all:
        predictions = [gg[0] for gg in predictions]
    return predictions

def write_props_format_predictions(out, predictions, gold_pred):
    '''
    Predictions contain a dictionary for example:
        [{'predicates':[],
        'roles':[[],[]...],
        'tokens':[],
        'tokens_ids':[],
        'pred_lin':[],
        'status': ParsedStatus}, {} ]
    Args:
        out: path of the output file
        predictions: list of dicts
    Returns:
    '''
    with open(Path(out), "w") as outfile:
        for i, pred in enumerate(predictions):
            tmp_role = ["*"]*len([x for x in gold_pred[i] if x!="-"])
            current_in_gold = 0
            for idx, p in enumerate(pred['predicates']):
                rol = pred['roles'][idx]
                pred_in_input = gold_pred[i][idx]
                if len(rol)>0:
                    outfile.write("{}\t{}\n".format(pred_in_input, "\t".join(rol)))
                    current_in_gold += 1
                else:
                    if pred_in_input!="-":
                        curr_role = copy.deepcopy(tmp_role)
                        curr_role[current_in_gold] = "(V*)"
                        outfile.write("{}\t{}\n".format(pred_in_input, "\t".join(curr_role)))
                        current_in_gold+=1
                    else:
                        outfile.write("{}\t{}\n".format(pred_in_input, "\t".join(tmp_role)))
            outfile.write("\n")


def overwrite_gold_with_predicted(out, predictions, gold_pred):
    sentence_id = 0
    print(len(predictions))
    with open(gold_pred, "r") as infile, open(out, "w") as outfile:
        for line_nr, line in enumerate(infile.readlines()):
            if line.startswith("#"): continue
            if line.rstrip()=="":
                sentence_id+=1
                outfile.write("\n")
            else:
                fields = line.rstrip().split("\t")
                pred_token_id = int(fields[0])-1
                curr_pred = predictions[sentence_id]
                new_line = copy.deepcopy(fields)
                gpredicate = fields[13]
                ppredicate = curr_pred['predicates'][pred_token_id]
                if ppredicate!="-":
                    ppredicate=ppredicate[:-3]+"."+ppredicate[-2:]
                else:
                    ppredicate="_"
                new_line[13] = ppredicate
                groles = fields[14:]
                proles = curr_pred['roles'][pred_token_id]
                for i in range(len(groles)):
                    if i<len(proles):
                        if proles[i] in ["*", "(V*)", "(N*)","", "*)"]:
                            formatted_role="_"
                        else:
                            prole_a = proles[i]
                            if prole_a.endswith(")"):
                                formatted_role = prole_a[1:-2].replace('ARG','A')
                                if formatted_role=="V": formatted_role="_"
                            else:
                                formatted_role = prole_a[1:-1].replace('ARG','A')
                                if formatted_role == "V": formatted_role = "_"
                        new_line[14+i] = formatted_role
                    else:
                        new_line[14+i]="_"
                outfile.write("{}\n".format("\t".join(new_line)))
        outfile.write("\n")

def compute_span_srl_f1(gold_roles, pred_roles):
    if not os.path.exists("out"): os.makedirs("out")
    subprocess.call(["perl scripts/srl-eval.pl {} {} > out/out_prl_test.txt".format(gold_roles, pred_roles)], shell=True)
    f1_score = float(open("out/out_prl_test.txt","r").readlines()[6].rstrip().split()[-1])
    return f1_score

def compute_dep_srl_f1(gold_roles, pred_roles, eval_name=""):
    if not os.path.exists("out"): os.makedirs("out")
    subprocess.call(["perl scripts/eval09.pl -q -g {} -s {} > out/out_dep_prl_test{}.txt".format(gold_roles, pred_roles, eval_name)],
                    shell=True)
    f = 0
    with open("out/out_dep_prl_test{}.txt".format(eval_name), "r") as infile:
        for line in infile:
            if "Labeled F1:" in line:
                f = line.rstrip().split()[-1][:-1]
                break
    f1_score =float(f)
    return f1_score
