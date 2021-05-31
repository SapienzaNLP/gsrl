import re
import torch
import spacy
import logging
from glob import glob
from pathlib import Path
from transformers import AutoConfig, BartTokenizer
from src.modelling.dataset import StructureDatasetTokenBatcherAndLoader, SRLDataset
from src.modelling.modeling_bart import CustomBartForConditionalGeneration
from src.modelling.tokenization_bart import SRLBARTTokenizer

from spacy.util import compile_infix_regex
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS


def instantiate_model_and_tokenizer(
        name=None,
        checkpoint=None,
        dropout = 0.15,
        attention_dropout = 0.15,
        from_pretrained = True,
        task="srl",
        task_type="span"
):

    if name is None:
        name = 'facebook/bart-large'

    if name == 'facebook/bart-base':
        tokenizer_name = 'facebook/bart-large'
    else:
        tokenizer_name = name

    config = AutoConfig.from_pretrained(name)
    config.output_past = False
    config.no_repeat_ngram_size = 0
    config.prefix = " "
    config.output_attentions = True
    config.dropout = dropout
    config.attention_dropout = attention_dropout


    if task=="srl":
        tokenizer = SRLBARTTokenizer.from_pretrained(
            tokenizer_name,
            config=config,
            task_type=task_type,
        )
    else:
        tokenizer = BartTokenizer.from_pretrained(
            tokenizer_name,
            config=config)

    if from_pretrained:
        model = CustomBartForConditionalGeneration.from_pretrained(name, config=config)
    else:
        model = CustomBartForConditionalGeneration(config)

    model.resize_token_embeddings(len(tokenizer.encoder))

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'])

    return model, tokenizer


def instantiate_loader(
        glob_pattn,
        tokenizer,
        task="srl",
        task_type="span",
        batch_size=500,
        evaluation=True,
        out=None,
        remove_longer_than=None,
        duplicate_per_predicate=False,
        identify_predicate=False,
):
    paths = []
    if isinstance(glob_pattn, str) or isinstance(glob_pattn, Path):
        glob_pattn = [glob_pattn]
    for gpattn in glob_pattn:
        paths += [Path(p) for p in glob(gpattn)]
    paths=sorted(paths)

    if task =="srl":
        dataset = SRLDataset(
            paths,
            tokenizer,
            remove_longer_than=remove_longer_than,
            duplicate_per_predicate=duplicate_per_predicate,
            task_type=task_type,
            identify_predicate=identify_predicate
        )
    else:
        raise ValueError

    loader = StructureDatasetTokenBatcherAndLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not evaluation,
    )
    gold_pred=None
    if evaluation:
        assert out is not None
        if task == "srl" and task_type=="span":
            if not duplicate_per_predicate:
                gold_pred = write_props_format(out, paths)
            else:
                gold_pred = write_duplicate_props_format(out, dataset)
        elif task=="srl" and task_type=="dep":
            if duplicate_per_predicate:
                write_duplicate_conll_format(out, paths, dataset)
            else:
                Path(out).write_text('\n\n'.join([p.read_text() for p in paths]))

    if evaluation:
        return loader, gold_pred
    else:
        return loader

def write_props_format(out, paths):
    total_sentences = 0
    sentences = []
    current_sent = []
    with open(Path(out), "w") as outfile:
        for path_ in paths:
            for path in glob(str(path_)):
                path = Path(path)
                with open(path, "r", encoding="utf-8") as infile:
                    discarded=False
                    for line in infile:
                        if line.startswith("#"): continue
                        fields = line.split()
                        if fields:
                            tok_id, tok, pos, pred, f, ner, coref = \
                                fields[2], fields[3], fields[4], fields[6], fields[7], fields[10], fields[-1]
                            if len(fields) > 12:
                                rol = fields[11:-1]
                            else:
                                rol = []
                                discarded = True
                                continue
                            p = pred if f!="-" else "-"
                            current_sent.append(p)
                            if len(rol)>0:
                                outfile.write("{}\t{}\n".format(p, "\t".join(rol)))
                            else:
                                outfile.write("{}\n".format(p))
                        else:
                            if discarded:
                                discarded=False
                                current_sent=[]
                            else:
                                sentences.append(current_sent)
                                outfile.write("\n")
                                total_sentences+=1
                                current_sent=[]
                    else:
                        outfile.write("\n")
        logging.warning("Total loaded sentences: {} {}".format(total_sentences, len(sentences)))
        return sentences

def write_duplicate_props_format(out, corpus):
    sentences_preds = []
    with open(Path(out), "w") as outfile:
        for sentence in corpus:
            sentence_tokens = sentence['structure']['sentence'].split()
            sentence_roles = sentence['structure']['roles']
            unroll_roles = ["*"]*len(sentence_tokens)
            unroll_predicates = ["-"]*len(sentence_tokens)
            assert len(sentence_roles)==1, "Something is wrong with duplicating sentences per predicate"
            for role in sentence_roles:
                unroll_predicates[role]=sentence_roles[role]['pred']
                for argument in sentence_roles[role]['arguments']:
                    start, end, label = argument
                    if end - start == 1:
                        unroll_roles[start]="({}*)".format(label.replace(":",""))
                    else:
                        unroll_roles[start] = "({}*".format(label.replace(":",""))
                        unroll_roles[end - 1] = "*)"

            for idx, p in enumerate(unroll_predicates):
                outfile.write("{}\t{}\n".format(p, unroll_roles[idx]))
            outfile.write("\n")
            sentences_preds.append(unroll_predicates)
    return sentences_preds

def write_duplicate_conll_format(out, paths, dataset):
    gold_file_sentences = []
    sentence_lines = []
    for path_ in paths:
        for path in glob(str(path_)):
            path = Path(path)
            with open(path, "r", encoding="utf-8") as infile:
                for line in infile:
                    if line.startswith("#"): continue
                    if line.rstrip()=="":
                        if len(sentence_lines)>0:
                            if len(sentence_lines[0].split("\t"))>14:
                                gold_file_sentences.append(sentence_lines)
                            sentence_lines = []
                    else:
                        sentence_lines.append(line.rstrip())
                if len(sentence_lines) > 0:
                    if len(sentence_lines[0].split("\t")) > 14:
                        gold_file_sentences.append(sentence_lines)
                    sentence_lines = []

    gold_file_index = 0
    initial_sentence = " ".join([x.split("\t")[1] for x in gold_file_sentences[0]])
    with open(Path(out), "w") as outfile:
        for sentence in dataset:
            if sentence['structure']['sentence']!=initial_sentence:
                gold_file_index+=1
            initial_sentence = " ".join([x.split("\t")[1] for x in gold_file_sentences[gold_file_index]])
            sentence_tokens = sentence['structure']['sentence'].split()
            sentence_roles = sentence['structure']['roles']
            unroll_roles = ["_"]*len(sentence_tokens)
            unroll_predicates = ["_"]*len(sentence_tokens)
            assert len(sentence_roles)==1, "Something is wrong with duplicating sentences per predicate"

            for role in sentence_roles:
                unroll_predicates[role]=sentence_roles[role]['pred']+"."+sentence_roles[role]['frameid']
                for argument in sentence_roles[role]['arguments']:
                    start, end, label = argument
                    assert end-start<=1, "Something is wrong, longer span in dep-srl"
                    c_role = "{}".format(label.replace(":", "").replace("ARG", "A"))
                    if c_role=="V" or c_role=="N":
                        c_role="_"
                    unroll_roles[start]=c_role

            for idx, ppredicate in enumerate(unroll_predicates):
                if ppredicate!="_":
                    p = "Y"
                else:
                    p="_"
                outfile.write("{}\t{}\t{}\t{}\n".format("\t".join(gold_file_sentences[gold_file_index][idx].split("\t")[:12]), p, ppredicate, unroll_roles[idx]))
            outfile.write("\n")

def load_nlp_pipeline():
    nlp = spacy.load("en_core_web_sm")
    # modify tokenizer infix patterns
    infixes = (
            LIST_ELLIPSES
            + LIST_ICONS
            + [
                r"(?<=[0-9])[+\-\*^](?=[0-9-])",
                r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                    al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
                ),
                r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
                # EDIT: commented out regex that splits on hyphens between letters:
                # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
                # EDIT: added regex that splits on ' between letters
                r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h="'"),
                r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
            ]
    )

    infix_re = compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_re.finditer
    return nlp

