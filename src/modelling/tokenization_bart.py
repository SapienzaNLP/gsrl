import os
import sys
import copy
import math
import enum
import torch
import string
import nltk
nltk.download("wordnet")
import regex as re
from pathlib import Path
from src.modelling import ROOT
from transformers import BartTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from src.modelling.linearization import CustomTokens

class ParsedStatus(enum.Enum):
    OK = 0
    FIXED = 1
    BACKOFF = 2


class CustomBartTokenizer(BartTokenizer):

    INIT = 'Ġ'

    ADDITIONAL = [CustomTokens.LIT_START, CustomTokens.LIT_END, CustomTokens.PRED]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patterns = re.compile(
            r""" ?<[a-z]+:?\d*>| ?:[^\s]+|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    @classmethod
    def from_pretrained(cls, pretrained_model_path, pred_min=5, task_type="span", *args, **kwargs):
        inst = super().from_pretrained(pretrained_model_path, *args, **kwargs)
        inst.task_type = task_type
        inst.all_role_labels = [x.rstrip().split("\t")[0] for x in open("data/vocab/srl_roles.txt")]
        inst.all_predicates = [x.rstrip().split("\t")[0] for x in open("data/vocab/srl_predicates.txt")]
        if task_type=="dep":
            inst.all_role_labels.extend([x.rstrip().split("\t")[0] for x in open("data/vocab/dep_srl_roles.txt") if
                                         x.rstrip().split("\t")[0] not in inst.all_role_labels])
            inst.all_predicates.extend([x.rstrip().split("\t")[0] for x in open("data/vocab/dep_srl_predicates.txt") if
                                    x.rstrip().split("\t")[0] not in set(inst.all_predicates)])
        #Init according to task or joint
        inst.init_custom_vocabulary(pred_min=pred_min, task_type=task_type)
        return inst

    def init_custom_vocabulary(self, pred_min=5,  task_type="span"):
        for tok in [self.bos_token, self.eos_token, self.pad_token, '<mask>', '<unk>']:
            ntok = self.INIT + tok
            i = self.encoder[tok]
            self.decoder[i] = ntok
            del self.encoder[tok]
            self.encoder[ntok] = i

        tokens = []
        srl_plus_predicates = dict()
        for line in Path(ROOT / 'data/vocab/predicates.txt').read_text().strip().splitlines():
            tok, count = line.split()
            if tok not in srl_plus_predicates:
                srl_plus_predicates[tok] = int(count)
            else:
                srl_plus_predicates[tok] += int(count)

        for line in Path(ROOT / 'data/vocab/srl_predicates.txt').read_text().strip().splitlines():
            tok, count = line.split()
            if tok not in srl_plus_predicates:
                srl_plus_predicates[tok] = int(count)
            else:
                srl_plus_predicates[tok] += int(count)

        if task_type=="dep":
            for line in Path(ROOT / 'data/vocab/dep_srl_predicates.txt').read_text().strip().splitlines():
                tok, count = line.split()
                if tok not in srl_plus_predicates:
                    srl_plus_predicates[tok] = int(count)
                else:
                    srl_plus_predicates[tok] += int(count)

        for tok in srl_plus_predicates:
            if srl_plus_predicates[tok]>=pred_min:
                tokens.append(tok)


        for tok in Path(ROOT / 'data/vocab/additions.txt').read_text().strip().splitlines():
            tokens.append(tok)

        for line in Path(ROOT / 'data/vocab/srl_roles.txt').read_text().strip().splitlines():
            tok, count = line.split()
            if tok not in set(tokens):
                tokens.append(tok)

        if task_type=="dep":
            for line in Path(ROOT / 'data/vocab/dep_srl_roles.txt').read_text().strip().splitlines():
                tok, count = line.split()
                if tok not in set(tokens):
                    tokens.append(tok)


        for cnt in range(50):
            #f"<P{cnt}>"
            tokens.append(CustomTokens.PRED.format(cnt))


        tokens += self.ADDITIONAL
        tokens = [self.INIT + t if t[0] not in ('_', '-') else t for t in tokens]
        tokens = [t for t in tokens if t not in self.encoder]
        self.old_enc_size = old_enc_size = len(self.encoder)
        for i, t in enumerate(tokens, start=old_enc_size):
            self.encoder[t] = i

        self.encoder = {k: i for i, (k, v) in enumerate(sorted(self.encoder.items(), key=lambda x: x[1]))}
        self.decoder = {v: k for k, v in sorted(self.encoder.items(), key=lambda x: x[1])}
        self.modified = len(tokens)

        self.bos_token = self.INIT + '<s>'
        self.pad_token = self.INIT + '<pad>'
        self.eos_token = self.INIT + '</s>'
        self.unk_token = self.INIT + '<unk>'
        special_preds = list()
        for cnt in range(50):
            special_preds.append(self.INIT + CustomTokens.PRED.format(cnt))
        self._additional_special_tokens=special_preds

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output
        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def _tok_bpe(self, token, add_space=True):
        # if add_space:
        #     token = ' ' + token.lstrip()
        tokk = []
        tok = token.strip()
        for tok in self.patterns.findall(' ' + token):
            tok = "".join(
                self.byte_encoder[b] for b in tok.encode("utf-8"))
            toks = self.bpe(tok).split(' ')
            tokk.extend(toks)
        return tokk

    def batch_encode_sentences(self, sentences, device=torch.device('cpu')):
        sentences = [s for s in sentences]
        extra = {'sentences': sentences}
        batch = super().batch_encode_plus(sentences, return_tensors='pt', pad_to_max_length=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        return batch, extra


class SRLBARTTokenizer(CustomBartTokenizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linearizer = None
        self.remove_pars = False

    def linearize(self, sentence_inst, identify_predicate=False):
        tokens, token_uni_ids, linearized_input = self.tokenize_srl(sentence_inst, identify_predicate)
        extra = {'linearized_structure': tokens, 'structure': sentence_inst, 'linearized_pred_sent': linearized_input}
        if token_uni_ids[-1] != (self.INIT + CustomTokens.EOS_N):
            tokens.append(self.INIT + CustomTokens.EOS_N)
            token_uni_ids.append(self.eos_token_id)

        return token_uni_ids, extra, linearized_input

    def tokenize_srl(self, sentence_inst, identify_predicate):
        linearized_input, linearized_srl = self.linearize_srl(sentence_inst['sentence'], sentence_inst['roles'], identify_predicate)
        linearized = re.sub(r"\s+", ' ', linearized_srl)
        bpe_tokens, bpe_token_ids = self.tokenize_custom(linearized)
        return bpe_tokens, bpe_token_ids, linearized_input

    def tokenize_custom(self, srl_sent):
        linearized_nodes = [CustomTokens.BOS_N] + srl_sent.split(' ')
        bpe_tokens = []

        for i,  tokk in enumerate(linearized_nodes):
            is_in_enc = self.INIT + tokk in self.encoder
            is_rel = tokk.startswith(':') and len(tokk) > 1
            is_spc = tokk.startswith('<') and tokk.endswith('>')
            is_of = tokk.startswith(':') and tokk.endswith('-of')
            is_frame = re.match(r'.+-\d\d', tokk) is not None

            if (is_rel or is_spc or is_frame or is_of):
                if is_in_enc:
                    bpe_toks = [self.INIT + tokk]
                elif is_frame:
                    bpe_toks = self._tok_bpe(tokk[:-3], add_space=True) + [tokk[-3:]]
                elif is_of:
                    rel = tokk[:-3]
                    if self.INIT + rel in self.encoder:
                        bpe_toks = [self.INIT + rel, '-of']
                    else:
                        bpe_toks = [self.INIT + ':'] + self._tok_bpe(rel[1:], add_space=True) + ['-of']
                elif is_rel:
                    bpe_toks = [self.INIT + ':'] + self._tok_bpe(tokk[1:], add_space=True)
                else:
                    bpe_toks = self._tok_bpe(tokk, add_space=True)

            else:
                if is_in_enc:
                    bpe_toks = [self.INIT + tokk]
                else:
                    bpe_toks = self._tok_bpe(tokk, add_space=True)

            bpe_tokens.append(bpe_toks)

        bpe_tokens = [b for bb in bpe_tokens for b in bb]
        bpe_token_ids = [self.encoder.get(b, self.unk_token_id) for b in bpe_tokens]

        return bpe_tokens, bpe_token_ids

    def linearize_srl(self, sentence, sentence_roles, identify_predicate):
        linearize_in = list()
        linearize_out = list()
        sentence_dict_in = {i: token for i, token in enumerate(sentence.split())}
        sentence_dict_out = {i: token for i, token in enumerate(sentence.split())}
        arg_dict = dict()
        total_preds = -1

        for i, el in enumerate(sentence.split()):
            if sentence_roles is None: continue
            if i in sentence_roles:
                total_preds += 1
                for arg in sentence_roles[i]['arguments']:
                    new_arg = (arg[0], arg[1], CustomTokens.PRED.format(total_preds) + " " + arg[2], True if sentence_roles[i]['noun_predicate'] else False)
                    if arg not in arg_dict:
                        arg_dict[new_arg] = dict()
                    arg_dict[new_arg]["nr_tokens"] = arg[1] - arg[0]
                    arg_dict[new_arg]["pred_id"] = i
                    arg_dict[new_arg]["pred"] = sentence_roles[i]["pred"]
                    arg_dict[new_arg]["frame"] = sentence_roles[i]["frameid"]

        for arg in sorted(arg_dict, key=lambda x: arg_dict[x]['nr_tokens']):
            start, end, label, noun_predicate = arg
            if label[-2:] == ":V" or (noun_predicate and label[-2:] == ":N"):
                text = arg_dict[arg]['pred'] + "-" + arg_dict[arg]["frame"]
            else:
                text = None
            if end - start == 1:
                text = sentence_dict_out[start] if text is None else text
                sentence_dict_out[start] = "{} [ {} ]".format(label, text)
                if label[-2:] == ":V" or (noun_predicate and label[-2:] == ":N"):
                    if noun_predicate:
                        label = label.split()[0]+" "+":N"
                    if not identify_predicate:
                        sentence_dict_in[start] = "{} [ {} ]".format(label, sentence_dict_in[start])

            else:
                end = end - 1
                text = sentence_dict_out[start] if text is None else text
                sentence_dict_out[start] = "{} [ {}".format(label, text)
                sentence_dict_out[end] = "{} ]".format(sentence_dict_out[end])
                if label[-2:] == ":V" or (noun_predicate and label[-2:] == ":N"):
                    if noun_predicate:
                        label = label.split()[0] + " " + ":N"
                    if not identify_predicate:
                        sentence_dict_in[start] = "{} [ {}".format(label, sentence_dict_in[start])
                        sentence_dict_in[end] = "{} ]".format(sentence_dict_in[end])

        for i in range(len(sentence_dict_out)):
            linearize_out.append(sentence_dict_out[i])

        for i in range(len(sentence_dict_in)):
            linearize_in.append(sentence_dict_in[i])

        return " ".join(linearize_in), " ".join(linearize_out)

    def batch_encode_structure(self, srl_insts, device=torch.device('cpu')):
        linearized, extras, srl_pred_inputs = zip(*[self.linearize(srl) for srl in srl_insts])
        return self.batch_encode_structure_from_linearized(linearized, extras, device=device)

    def batch_encode_structure_from_linearized(self, linearized, extras=None, device=torch.device('cpu')):
        if extras is not None:
            batch_extra = {'linearized_structure': [], 'structure': []}
            for extra in extras:
                batch_extra['structure'].append(extra['structure'])
                batch_extra['linearized_structure'].append(extra['linearized_structure'])
        else:
            batch_extra = {}
        maxlen = 0
        batch = []
        for token_uni_ids in linearized:
            maxlen = max(len(token_uni_ids), maxlen)
            batch.append(token_uni_ids)
        batch = [x + [self.pad_token_id] * (maxlen - len(x)) for x in batch]
        batch = torch.tensor(batch).to(device)
        batch = {'decoder_input_ids': batch[:, :-1], 'lm_labels': batch[:, 1:]}
        return batch, batch_extra

    def decode_srl(self, tokens, input_sent, gold_linearization="", nlp=None, task_type="span"):
        try:
            decoded_tokens = self.decode(tokens)
            instance, decoded_tokens = self._decode_srl(decoded_tokens, input_sent,gold_linearization, nlp, task_type)
            status =ParsedStatus.OK
        except Exception as e:
            print('Decoding failure: {}'.format(e), file=sys.stderr)
            instance = self._dumb_decode_srl(input_sent)
            status = ParsedStatus.BACKOFF
            decoded_tokens = "<empty>"

        return status, instance, decoded_tokens

    def _decode_srl(self, decoded_tokens, input_sent, gold_linearization, nlp, task_type):

        def _extract_pred_id(predicates):
            pred2nr = dict()
            pred2nr_sorted = dict()
            for pred in predicates:
                predicate_nr = int(pred[2:-1])
                pred2nr[pred] = predicate_nr

            for i, pred in enumerate(sorted(pred2nr.items(), key=lambda item: item[1])):
                pred2nr_sorted[pred[0]] = i

            return pred2nr_sorted

        def _extract_pred_type(noun_predicates, pred_order):
            pred2type=dict()
            for np in noun_predicates:
                pred = np.split()[0]
                curr_pred_order = pred_order[pred]
                pred2type[curr_pred_order] = "N"
            for p in pred_order:
                if pred_order[p] not in pred2type:
                    pred2type[pred_order[p]]="V"

            return pred2type

        def fix_spaces_rules(decoded_tokens): #Fixing tokenization problems
            decoded_tokens = decoded_tokens.replace("< ", "<").replace(" >", ">").replace("< / ", "</").replace("` `","``").replace("/ ?","/?")
            decoded_tokens = decoded_tokens.replace(": N",":N").replace(": V", ":V").replace(": ARG", ":ARG").replace(": R-ARG", ":R-ARG").replace(": C-ARG", ":C-ARG")
            decoded_tokens = decoded_tokens.replace("n ' t", "n't").replace("' s","'s").replace("' ve", "'ve").replace("' m","'m")
            decoded_tokens = decoded_tokens.replace("a .m .","a.m.").replace("a. m.", "a .m .").replace("'-RRB-","' -RRB-").replace("{ one ? }", "{one?}")
            decoded_tokens = decoded_tokens.replace(". . . .", ".. ..").replace(".. .", "...").replace(". ..","...").replace("* * * *", "****")
            decoded_tokens = decoded_tokens.replace(" . com", ".com").replace(". COM",".COM").replace("% -","%-")
            decoded_tokens = decoded_tokens.replace("+ ","+").replace("% uh", "%uh").replace("% um", "%um").replace("% ah", "%ah")
            decoded_tokens = re.sub(r'(\d)\s*(\.)\s*(\d)', r'\1\2\3', decoded_tokens)
            decoded_tokens = re.sub(r"(\d)(\')", r'\1 \2', decoded_tokens)
            decoded_tokens = re.sub(r'(\.)\s*(\d)', r'\1\2', decoded_tokens)
            decoded_tokens = re.sub(r'([a-zA-Z])\s?\.\s?([aA-zZ])\s?\.', r'\1.\2.', decoded_tokens)

            return decoded_tokens

        decoded_tokens = " ".join([x for x in decoded_tokens.split() if x!="<pad>"]).strip()
        decoded_tokens = re.sub(r'([\].])(.)', r'\1 \2', decoded_tokens)
        if nlp is not None:
            decoded_tokens = " ".join([token.text for token in nlp(decoded_tokens)])
            decoded_tokens = fix_spaces_rules(decoded_tokens)

        one_bracket = abs(decoded_tokens.count('[')-decoded_tokens.count(']'))==1

        assert decoded_tokens.count('[') == decoded_tokens.count(']') or one_bracket, 'Square brackets do not match'
        instance = dict()
        predicates = re.findall(r'<P\d+> :V \[ [a-z\%]+-\d\d \]', decoded_tokens)
        noun_predicates = re.findall(r'<P\d+> :N \[ [a-z\%]+-\d\d \]', decoded_tokens)
        predicate_order = _extract_pred_id(set(re.findall(r'<P\d+>',decoded_tokens)))
        predicate_order2type =_extract_pred_type(noun_predicates, predicate_order)
        predicates.extend(noun_predicates)
        roles = []
        tokens = []
        token_ids = []
        predicate_list = []

        tokid = -1
        pending_role="*"
        decoded_tokens = decoded_tokens.split()
        input_tokens = input_sent.split()
        pending = list()
        pending_label = list()
        pending_pred_id = list()
        for i, token in enumerate(decoded_tokens):
            if token =="<s>": continue
            if token =="</s>":
                break
            if token=="<pad>":
                break
            # clean token
            if re.match(r'<P\d+>', token) is None and token not in self.all_role_labels and token not in ["[","]"]:
                if len(pending)==0:
                    tokid+=1
                    token_ids.append(tokid)
                    tokens.append(token)
                    predicate_list.append("-")
                    role_to = []
                    for r in range(len(predicates)):
                        role_to.append("*")
                    roles.append(role_to)
                else:
                    tokid += 1
                    token_ids.append(tokid)
                    tokens.append(token)
                    role_to = []
                    total_roles_per_token = copy.deepcopy(pending)
                    for k in range(len(total_roles_per_token)):
                        try:
                            label, pred_nr = pending_label.pop()
                        except Exception as e:
                            raise ValueError("{} pending_label".format(e))
                        for r in range(len(predicates)):
                            if r == pred_nr:
                                if k == 0:
                                    role_to.append(label)
                                else:
                                    if role_to[r]!="*" and label=="(N*":
                                        role_to[r] = role_to[r]
                                    else:
                                        role_to[r] = label
                            else:
                                if k==0:
                                    role_to.append("*")
                                else:
                                    role_to[r] = role_to[r]
                        try:
                            pending_el = pending.pop()
                        except Exception as e:
                            raise ValueError("{} pending_el".format(e))
                    roles.append(role_to)
                    if len(total_roles_per_token)>1:
                        if "predicate" in total_roles_per_token:
                            if task_type =="dep":
                                predicate_list.append(token)
                            else:
                                predicate_list.append(token.split("-")[0])
                        else:
                            predicate_list.append('-')
                    else:
                        if pending_el == "predicate":
                            if task_type =="dep":
                                predicate_list.append(token)
                            else:
                                predicate_list.append(token.split("-")[0])
                        elif pending_el == "role":
                            predicate_list.append("-")

            # tag token
            elif re.match(r'<P\d+>', token) is not None:
                predicate_nr = predicate_order[token]
                if i+1<len(decoded_tokens):
                    if decoded_tokens[i+1]==":V":
                        pending.append("predicate")
                        pending_role = "(V*"
                    elif decoded_tokens[i+1]==":N":
                        pending.append("predicate")
                        pending_role = "(N*"
                    elif decoded_tokens[i+1] in self.all_role_labels:
                        pending.append("role")
                        pending_role = decoded_tokens[i+1].replace(":","(")+"*"
                    pending_label.append((pending_role, predicate_nr))
                    pending_pred_id.append(predicate_nr)

            elif token == "]" and len(pending_pred_id)>0:
                if decoded_tokens[i-1]!="[":
                    role_to = []
                    pred_nr = pending_pred_id.pop()
                    for r in range(len(predicates)):
                        old_role = roles[tokid][r]
                        if r == pred_nr:
                            if not old_role.endswith(")"):
                                role_to.append(old_role+")")
                            else:
                                if self.task_type=="span":
                                    raise ValueError("Old role already is closed in prediction --> {}".format(old_role))
                                else:
                                    role_to.append(old_role)
                        else:
                            role_to.append(old_role)
                    roles[tokid] = role_to


        def get_predicate_pos(roles, pred_id, role=None):
            try:
                roles_per_pred = [role[pred_id] for role in roles]
                if role is None:
                    role = unclosed_roles(roles_per_pred)
                    if role is None:
                        return None, None
                    if role is False:
                        return False, False
                pred_pos = roles_per_pred.index(role)
                return pred_pos, roles_per_pred
            except:
                return None, None

        def unclosed_roles(roles, status=""):
            unclosed = []
            for r in roles:
                if r not in ["*","*)"] and not r.endswith(")"):
                    unclosed.append(r)
                elif r == "*)":
                    if len(unclosed)>0:
                        unclosed.pop()
                    else:
                        raise ValueError("Closed before opened {}".format(status))
            if len(unclosed)>0:
                return unclosed[-1]
            else:
                return None

        if len(pending_pred_id) != 0 and one_bracket:
            all_pending_preds = copy.deepcopy(pending_pred_id)
            for k in range(len(all_pending_preds)):
                role_to = []
                try:
                    pred_nr = pending_pred_id.pop()
                except Exception as e:
                    raise ValueError("{} pending_pred_id one_bracket fix".format(e))
                predicate_position, roles_per_pred = get_predicate_pos(roles, pred_nr, "(V*)")
                #if predicate_position is False: return None
                unclosed_position, _ = get_predicate_pos(roles, pred_nr)
                #if unclosed_position is False: return None
                if unclosed_position is not None:
                    if task_type=="span":
                        if unclosed_position > predicate_position:
                            try:
                                idx = max(index for index, item in enumerate(roles_per_pred[:-1]) if item == "*")
                            except:
                                idx = unclosed_position
                        else:
                            idx = unclosed_position
                            for index, item in enumerate(roles_per_pred[unclosed_position+1:predicate_position]):
                                if item!="*":
                                    break
                                idx = max(idx, index)
                    else:
                        idx = unclosed_position

                    for r in range(len(predicates)):
                        old_role = roles[idx][r]
                        if r == pred_nr:
                            if old_role.endswith(")"):
                                if task_type == "span":
                                    raise ValueError("Old role already is closed in fix --> {}".format(old_role))
                                else:
                                    role_to.append(old_role)
                            else:
                                role_to.append(old_role + ")")
                        else:
                            role_to.append(old_role)
                    roles[idx] = role_to
        if task_type=="span":
            for pred_id in range(len(predicates)):
                unclosed = unclosed_roles([role[pred_id] for role in roles])
                if unclosed is not None:
                    #return None
                    raise ValueError("Before fixing length unclosed = {}".format(unclosed))

        if len(tokens) == len(input_tokens):
            instance["predicates"] = predicate_list
            instance["roles"] = roles
            instance["tokens"] = tokens
            instance['token_ids'] = token_ids
            return instance, decoded_tokens
        else:
            try:
                fix_pad_output = self.pad_missing_words(input_tokens, tokens, roles, predicate_list, task_type)
                tokens, token_ids, roles, predicate_list = fix_pad_output
            except Exception as e:
                assert len(tokens)==len(input_tokens), '{}: Different lengths for generated tokens: {} and input tokens: {}'.format(e, len(tokens),len(input_tokens))

            if task_type == "span":
                for pred_id in range(len(predicates)):
                    unclosed = unclosed_roles([role[pred_id] for role in roles], "after fix")
                    if unclosed is not None:
                        raise ValueError("After fixing length something happened with roles, unclosed = {}".format(unclosed))

            instance["predicates"] = predicate_list
            instance["roles"] = roles
            instance["tokens"] = tokens
            instance['token_ids'] = token_ids

            return instance, decoded_tokens

    def pad_missing_words(self, gold_l, pred_l, roles, predicate_list, task_type):
        stemmer = SnowballStemmer("english")
        lemmatizer = WordNetLemmatizer()
        fix_pred = []
        fix_token_ids = []
        fix_roles = []
        fix_predicate_list=[]
        tok_in_pred = 0
        skip_one = False
        for i, tok in enumerate(gold_l):
            if skip_one:
                skip_one = False
                continue
            if tok_in_pred<len(pred_l):
                curr_pred = pred_l[tok_in_pred]
                if tok == curr_pred:
                    fix_pred.append(curr_pred)
                    fix_predicate_list.append(predicate_list[tok_in_pred])
                    fix_roles.append(roles[tok_in_pred])
                    fix_token_ids.append(len(fix_token_ids))
                    tok_in_pred += 1
                elif re.match('.+-\d\d$', curr_pred) is not None and (stemmer.stem(curr_pred.split("-")[0]) == stemmer.stem(tok) or curr_pred.split("-")[0] == lemmatizer.lemmatize(tok, pos="v") or tok.startswith("'")):
                        fix_predicate_list.append(predicate_list[tok_in_pred])
                        fix_pred.append(curr_pred)
                        fix_roles.append(roles[tok_in_pred])
                        fix_token_ids.append(len(fix_token_ids))
                        tok_in_pred+=1
                else:
                    if len(pred_l) > tok_in_pred + 1 and curr_pred + pred_l[tok_in_pred + 1] == tok: #if we need to merge/remove space
                        fix_pred.append(curr_pred + pred_l[tok_in_pred + 1])
                        fix_token_ids.append(len(fix_token_ids))
                        if len(roles)>tok_in_pred+1:
                            if len(set(roles[tok_in_pred]))==1 and roles[tok_in_pred][0]=="*":
                                fix_roles.append(roles[tok_in_pred+1])
                            elif len(set(roles[tok_in_pred+1]))==1 and roles[tok_in_pred+1][0]=="*":
                                fix_roles.append(roles[tok_in_pred])
                            else:
                                new_role = []
                                for k in range(len(roles[tok_in_pred])):
                                    if roles[tok_in_pred][k]!="*" :
                                        if roles[tok_in_pred][k].endswith("*)"):
                                            new_role.append(roles[tok_in_pred][k])
                                        elif roles[tok_in_pred+1][k]=="*)":
                                            new_role.append(roles[tok_in_pred][k]+")")
                                        else:
                                            new_role.append(roles[tok_in_pred][k])
                                    else:
                                        new_role.append(roles[tok_in_pred+1][k])

                                fix_roles.append(new_role)
                            if predicate_list[tok_in_pred] != "-":
                                fix_predicate_list.append(predicate_list[tok_in_pred])
                            else:
                                fix_predicate_list.append(predicate_list[tok_in_pred + 1])
                        else:
                            fix_roles.append(roles[tok_in_pred])
                            fix_predicate_list.append(predicate_list[tok_in_pred])

                        tok_in_pred += 2
                        # here a pop at current position or next position is needed (if its not a label)
                    elif len(gold_l) > i + 1 and curr_pred == tok + gold_l[i + 1]: #if we need to add a space wrt to gold
                        fix_pred.append(tok)
                        fix_pred.append(gold_l[i + 1])
                        fix_predicate_list.append(predicate_list[tok_in_pred])
                        fix_roles.append(roles[tok_in_pred])
                        fix_token_ids.append(len(fix_token_ids))
                        fix_predicate_list.append("-")
                        fix_roles.append(["*"]*len(roles[tok_in_pred]))
                        fix_token_ids.append(len(fix_token_ids))
                        skip_one = True
                        tok_in_pred += 1
                    elif len(pred_l) > tok_in_pred + 1 and pred_l[tok_in_pred + 1] == tok: #if one token is extra but the next predicted is ok
                        fix_pred.append(pred_l[tok_in_pred + 1])
                        fix_token_ids.append(len(fix_token_ids))
                        if len(roles) > tok_in_pred + 1:
                            if len(set(roles[tok_in_pred])) == 1 and roles[tok_in_pred][0]=="*":
                                fix_roles.append(roles[tok_in_pred + 1])
                            elif len(set(roles[tok_in_pred + 1])) == 1 and roles[tok_in_pred+1][0]=="*":
                                fix_roles.append(roles[tok_in_pred])
                            else:
                                new_role = []
                                for k in range(len(roles[tok_in_pred])):
                                    if roles[tok_in_pred][k] != "*" and not roles[tok_in_pred][k].endswith("*)"):
                                        if roles[tok_in_pred + 1][k] == "*)":
                                            new_role.append(roles[tok_in_pred][k] + ")")
                                        else:
                                            new_role.append(roles[tok_in_pred][k])
                                    else:
                                        new_role.append(roles[tok_in_pred + 1][k])

                                fix_roles.append(new_role)
                            if predicate_list[tok_in_pred] != "-":
                                fix_predicate_list.append(predicate_list[tok_in_pred])
                            else:
                                fix_predicate_list.append(predicate_list[tok_in_pred + 1])
                        else:
                            fix_roles.append(roles[tok_in_pred])
                            fix_predicate_list.append(predicate_list[tok_in_pred])

                        tok_in_pred += 2
                        #continue #Maybe here to check ) in roles before ignoring
                    elif len(gold_l) > i + 1 and curr_pred == gold_l[i + 1]: #if one token is missing but next next is okay
                        fix_pred.append(gold_l[i])
                        fix_predicate_list.append("-")
                        fix_roles.append(["*"] * len(roles[tok_in_pred]))
                        fix_token_ids.append(len(fix_token_ids))
                    elif len(gold_l) > i + 1 and len(pred_l)>tok_in_pred+1 and pred_l[tok_in_pred+1] == gold_l[i + 1]: #if next tokens are same but current different
                        if re.match('.+-\d\d', curr_pred) is not None:
                            l = lemmatizer.lemmatize(gold_l[i], pos="v")
                            fix_pred.append(l)
                            if task_type=="dep":
                                fix_predicate_list.append(l+"-"+curr_pred.split("-")[-1])
                            else:
                                fix_predicate_list.append(l)
                            fix_roles.append(roles[tok_in_pred])
                        else:
                            fix_pred.append(gold_l[i])
                            fix_predicate_list.append("-")
                            fix_roles.append(["*"] * len(roles[tok_in_pred]))
                        fix_token_ids.append(len(fix_token_ids))
                        tok_in_pred+=1
                    else:
                        #where edit distance is bigger than 1
                        #pred *
                        buffer=[]
                        start = tok_in_pred
                        end = None
                        for p, candidate in enumerate(pred_l[tok_in_pred:]):
                            if len(gold_l)>i+1 and candidate == gold_l[i+1]:
                                end = p+tok_in_pred
                                break

                        if end is not None:
                            total_skipped = end - start
                            roles_of_skipped = roles[start:end]
                            predicates_of_skipped = predicate_list[start:end]
                            if len(set(predicates_of_skipped))==1 and predicates_of_skipped[0]=="-":
                                fix_predicate_list.append("-")
                            else:
                                fix_predicate_list.append([x for x in predicates_of_skipped if x!="-"][0])
                            new_role = []
                            for r in range(len(roles_of_skipped[0])):  #  [[*,	(ARG0*),	*], [*, V*, *], [*), *, * ]]
                                current_roles = [x[r] for x in roles_of_skipped]
                                if len(set(current_roles))==1:
                                    new_role.append(current_roles[0])
                                else:
                                    new_role.append([x for x in current_roles if x!="*"][0])

                            fix_pred.append(gold_l[i])
                            fix_roles.append(new_role)
                            fix_token_ids.append(len(fix_token_ids))
                            tok_in_pred+=total_skipped


                        else:
                            fix_pred.append(gold_l[i]) #**Tuftees**
                            fix_predicate_list.append("-")
                            fix_roles.append(["*"] * len(roles[tok_in_pred]))
                            fix_token_ids.append(len(fix_token_ids))
            else:
                fix_pred.append(gold_l[i])
                fix_predicate_list.append("-")
                fix_roles.append(["*"] * len(roles[-1]))
                fix_token_ids.append(len(fix_token_ids))

        if len(gold_l)!=len(fix_pred):
            raise ValueError("Expanding didn't work properly")
        return fix_pred, fix_token_ids, fix_roles, fix_predicate_list

    def _dumb_decode_srl(self, input_sent):

        input_tokens = input_sent.split()
        instance = dict()
        predicate_list =[]
        roles = []
        tokens = []
        token_ids = []
        for i in range(len(input_tokens)):
            predicate_list.append("-")
            roles.append([])
            tokens.append("<pad>")
            token_ids.append(i)

        instance["predicates"] = predicate_list
        instance["roles"] = roles
        instance["tokens"] = tokens
        instance['token_ids'] = token_ids

        return instance