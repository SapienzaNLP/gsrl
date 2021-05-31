import torch
import random
import logging
from cached_property import cached_property
from torch.utils.data import Dataset
from src.modelling.IO import read_span_srl_data, read_dep_srl_data

class SRLDataset(Dataset):

    def __init__(
            self,
            paths,
            tokenizer,
            device=torch.device('cpu'),
            remove_longer_than=None,
            duplicate_per_predicate=False,
            task_type="span",
            identify_predicate=False

    ):
        self.paths = paths
        self.tokenizer = tokenizer
        self.device = device

        if task_type=="span":
            corpus_all = read_span_srl_data(paths, duplicate_per_predicate)
        else:
            corpus_all = read_dep_srl_data(paths, duplicate_per_predicate)
        corpus = corpus_all.sentences
        logging.warning("Total sentences in corpus: {}".format(len(corpus)))
        self.srl_structures = []
        self.sentences = []
        self.linearized = []
        self.linearized_extra = []
        self.remove_longer_than = remove_longer_than
        self.identify_predicate = identify_predicate

        for instance in corpus:
            l, e, input_sentence_pred = self.tokenizer.linearize(instance, identify_predicate=identify_predicate)
            try:
                self.tokenizer.batch_encode_sentences([input_sentence_pred])
            except:
                logging.warning('Invalid sentence!')
                continue
            if remove_longer_than and len(l) > remove_longer_than:
                continue
            if len(l) > 1024:
                logging.warning('Sequence longer than 1024 included. BART does not support it!')

            self.sentences.append(input_sentence_pred)
            self.srl_structures.append(l)
            self.linearized.append(l)
            self.linearized_extra.append(e)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]
        if self.linearized is not None:
            sample['linearized_structure_ids'] = self.linearized[idx]
            sample.update(self.linearized_extra[idx])
        return sample

    def size(self, sample):
        return len(sample['linearized_structure_ids'])

    def collate_fn(self, samples, device=torch.device('cpu')):
        x = [s['sentences'] for s in samples]
        x, extra = self.tokenizer.batch_encode_sentences(x, device=device)
        if 'linearized_structure_ids' in samples[0]:
            y = [s['linearized_structure_ids'] for s in samples]
            y, extra_y = self.tokenizer.batch_encode_structure_from_linearized(y, samples, device=device)
            extra.update(extra_y)
        else:
            y = None
        extra['ids'] = [s['id'] for s in samples]
        return x, y, extra


class StructureDatasetTokenBatcherAndLoader:

    def __init__(self, dataset, batch_size=800, device=torch.device('cpu'), shuffle=False, sort=False):
        assert not (shuffle and sort)
        self.batch_size = batch_size
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.sort = sort

    def __iter__(self):
        it = self.sampler()
        it = ([[self.dataset[s] for s in b] for b in it])
        it = (self.dataset.collate_fn(b, device=self.device) for b in it)
        return it

    @cached_property
    def sort_ids(self):
        lengths = [len(s.split()) for s in self.dataset.sentences]
        ids, _ = zip(*sorted(enumerate(lengths), reverse=True))
        ids = list(ids)
        return ids

    def sampler(self):
        ids = list(range(len(self.dataset)))[::-1]

        if self.shuffle:
            random.shuffle(ids)
        if self.sort:
            ids = self.sort_ids.copy()

        batch_longest = 0
        batch_nexamps = 0
        batch_ntokens = 0
        batch_ids = []

        def discharge():
            nonlocal batch_longest
            nonlocal batch_nexamps
            nonlocal batch_ntokens
            ret = batch_ids.copy()
            batch_longest *= 0
            batch_nexamps *= 0
            batch_ntokens *= 0
            batch_ids[:] = []
            return ret

        while ids:
            idx = ids.pop()
            size = self.dataset.size(self.dataset[idx])
            cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
            if cand_batch_ntokens > self.batch_size and batch_ids:
                yield discharge()
            batch_longest = max(batch_longest, size)
            batch_nexamps += 1
            batch_ntokens = batch_longest * batch_nexamps
            batch_ids.append(idx)

            if len(batch_ids) == 1 and batch_ntokens > self.batch_size:
                yield discharge()

        if batch_ids:
            yield discharge()