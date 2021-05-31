import glob
from pathlib import Path
from typing import List, Union, Iterable

class SRL_Corpus():

    def __init__(self):
        self.sentences = []
        self.role_labels = []
        self.predicate_labels =[]

    def __len__(self):
        return len(self.sentences)

    def unroll_instance(self, instance):
        for el in instance["roles"]:
            new_instance = dict()
            new_instance['sentence'] = instance['sentence']
            new_instance['ners'] = instance['ners']
            new_instance['roles'] = dict()
            new_instance['roles'][el] = instance['roles'][el]
            self.sentences.append(new_instance)

    def populate(self, instance, role_labels, predicate_labels, duplicate_per_predicate, task_type="span"):
        #Discard sentences without any roles len(sentence['roles']) > 0
        #if instance['roles'] is None: return
        if instance['roles'] is None and (duplicate_per_predicate or task_type=="span"): return # and task_type=="span": return
        if not duplicate_per_predicate:
            self.sentences.append(instance)
        else:
            self.unroll_instance(instance)
        if role_labels!=None:
            self.role_labels.extend(role_labels)
        if predicate_labels!=None:
            self.predicate_labels.extend(predicate_labels)

class SRL_Instance():

    def __init__(self):
        self.token_ids =[]
        self.tokens =[]
        self.postags = []
        self.predicates = []
        self.frame_ids = []
        self.nertags = []
        self.roles = []
        self.corefs = []
        self.sentence = None
        self.linearization = None

    def set_fields(self, tok_id, tok, pos="*", pred="-", frame="-", ner="*", coref="*", rol=None):
        self.token_ids.append(tok_id)
        self.tokens.append(tok)
        self.postags.append(pos)
        self.predicates.append(pred)
        self.frame_ids.append(frame)
        self.nertags.append(ner)
        self.roles.append(rol)
        self.corefs.append(coref)

    def combine_fields(self, type="span"):
        self.sentence = dict()
        self.sentence['sentence'] = " ".join(self.tokens)

        merged_ = self.merge_roles(type)
        self.sentence['ners'] = self.merge_ners() if type=="span" else None

        if merged_ is not None:
            if len(self.roles[0]) > 0:
                self.sentence['roles'], roles_labels, predicate_labels = merged_
            else:
                self.sentence['roles'], roles_labels, predicate_labels = None, None, None
        else:
            return None

        return self.sentence, roles_labels, predicate_labels

    def merge_roles(self, type="span"):
        role_labels = list()
        predicate_labels = list()
        non_empty_predicate = [(i,x) for i, x in enumerate(self.predicates) if x!="-" and self.frame_ids[i]!="-"]

        if len(non_empty_predicate)!=len(self.roles[0]):
            return None

        predicate_structure = dict()
        for (pos, pred) in non_empty_predicate:
            predicate_structure[pos]=dict()
            predicate_structure[pos]["pred"] = pred
            predicate_structure[pos]["frameid"] = self.frame_ids[pos]
            predicate_structure[pos]["noun_predicate"] = False
            predicate_structure[pos]["arguments"] = list()
            predicate_labels.append('{}-{}'.format(pred.lower(), self.frame_ids[pos]))

        for i, (pos,pred) in enumerate(non_empty_predicate):
            if type=="dep":
                if self.roles[pos][i]=="*":
                    self.roles[pos][i] = "(V*)"
                else:
                    predicate_structure[pos]["noun_predicate"]=True
                    predicate_structure[pos]["arguments"].append((pos, pos+1, ":N"))
                    role_labels.append(":N")
            start = -1
            current_role = ""
            for j, roles in enumerate(self.roles):
                el = roles[i]
                if el!="*":
                    if el.startswith("("):
                        if el.endswith("*)"):
                            predicate_structure[pos]["arguments"].append((j, j + 1, ":"+el[1:-2]))
                            role_labels.append(":"+el[1:-2])
                            current_role = ""
                            start = -1
                        else:
                            current_role = el[1:-1]
                            start = j
                    elif el == "*)":
                        predicate_structure[pos]["arguments"].append((start, j+1, ":"+current_role))
                        role_labels.append(":"+current_role)
                        current_role = ""
                        start = -1

        return predicate_structure, role_labels, predicate_labels

    def merge_ners(self):
        sentence_ners = list()
        start=-1
        current_ner = ""
        for i, el in enumerate(self.nertags):
            if el!="*":
                if el.startswith("("):
                    if el.endswith(")"):
                        sentence_ners.append((i, i+1, el[1:-1]))
                        current_ner = ""
                        start = -1
                    else:
                        current_ner = el[1:-1]
                        start = i
                elif el=="*)":
                    sentence_ners.append((start, i+1, current_ner))
                    current_ner = ""
                    start = -1
        return sentence_ners

def read_span_srl_data(
    paths: List[Union[str, Path]], duplicate_for_predicate: bool
    ):
    #fields = ["domain", "sent_id","token_id",
    # "token", "pos", "dep", "predicate", "sense", "m1", "m2", "m3", "srl_annotations"]
    assert paths, "Provided data path does not exist"
    if not isinstance(paths, Iterable):
        paths = [paths]

    corpus = SRL_Corpus()
    corpus_continue = None
    for path_ in paths:
        for path in glob.glob(str(path_)):
            path = Path(path)
            with open(path, "r", encoding="utf-8") as infile:
                if corpus_continue is None:
                    sentence_fields = SRL_Instance()
                else:
                    sentence_fields = corpus_continue
                    corpus_continue = None
                for line in infile:
                    if line.startswith("#"): continue
                    fields = line.split()
                    if fields:
                        tok_id, tok, pos, pred, f, ner, coref = \
                            fields[2], fields[3], fields[4], fields[6], fields[7], fields[10], fields[-1]
                        if len(fields)>12:
                            rol = fields[11:-1]
                        else:
                            rol = []

                        sentence_fields.set_fields(tok_id, tok, pred=pred, rol=rol, pos=pos,  frame=f, ner=ner, coref=coref)
                    else:
                        combination = sentence_fields.combine_fields()
                        if combination is not None:
                            sentence, roles_labels, predicate_labels = combination
                            corpus.populate(sentence, roles_labels, predicate_labels, duplicate_for_predicate)
                        else:
                            print(path_)
                        sentence_fields = SRL_Instance()
                else:
                    corpus_continue = sentence_fields
    if sentence_fields is not None and len(sentence_fields.tokens) > 0:
        combination = sentence_fields.combine_fields()
        if combination is not None:
            sentence, roles_labels, predicate_labels = combination
            corpus.populate(sentence, roles_labels, predicate_labels, duplicate_for_predicate)
    return corpus

def read_dep_srl_data(paths: List[Union[str, Path]], duplicate_for_predicate: bool):
    # fields = ["tokid", "token","lemma","plemma","pos", "ppos",
    # "morph", "pmorph", "head", "phead", "deprel", "pdeprel",
    # "fillpred", "pred", "aperdn"]
    assert paths, "Provided data path does not exist" 
    if not isinstance(paths, Iterable):
        paths = [paths]

    corpus = SRL_Corpus()
    corpus_continue = None
    for path_ in paths:
        for path in glob.glob(str(path_)):
            path = Path(path)
            with open(path, "r", encoding="utf-8") as infile:
                if corpus_continue is None:
                    sentence_fields = SRL_Instance()
                else:
                    sentence_fields = corpus_continue
                    corpus_continue = None
                for line in infile:
                    if line.startswith("#"): continue
                    fields = line.split()
                    if fields:
                        tok_id, tok, pos, pred_f, pred_true = \
                            fields[0], fields[1], fields[4], fields[13], fields[12]
                        pred_f = pred_f.split(".") if pred_f!="_" else ["-","-"]
                        pred, f = pred_f[0], pred_f[1]
                        if len(fields) > 14:
                            roles = [x if x!="_" else "*" for x in fields[14:]]
                            rol = []
                            for r in roles:
                                if r.startswith("A"):
                                    rol.append("(ARG{}*)".format(r[1:]))
                                elif r.startswith("R-A"):
                                    rol.append("(R-ARG{}*)".format(r[3:]))
                                elif r.startswith("C-A"):
                                    rol.append("(C-ARG{}*)".format(r[3:]))
                                else:
                                    rol.append(r)
                        else:
                            rol=[]

                        sentence_fields.set_fields(tok_id, tok, pred=pred, rol=rol, pos=pos, frame=f)
                    else:
                        combination = sentence_fields.combine_fields(type="dep")
                        if combination is not None:
                            sentence, roles_labels, predicate_labels = combination
                            corpus.populate(sentence, roles_labels, predicate_labels, duplicate_for_predicate, task_type="dep")
                        else:
                            print(path_)
                        sentence_fields = SRL_Instance()
                else:
                    corpus_continue = sentence_fields
        if sentence_fields is not None and len(sentence_fields.tokens)>0:
            combination = sentence_fields.combine_fields(type="dep")
            if combination is not None:
                sentence, roles_labels, predicate_labels = combination
                corpus.populate(sentence, roles_labels, predicate_labels, duplicate_for_predicate, task_type="dep")

    return corpus

