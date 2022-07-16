from datasets import load_dataset
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
import pytorch_lightning as pl


def yahoo_answers_dataset(seed: int = 42, train_num: int = 200):
    """
    seed: seed value to be used for random initialisation
    val_ratio: number of samples to be kept as training data
    tok: String Identifier for tokenizer Ex. 'bert-base-uncased'
    truncation : If set to True, sequence will be truncated to max_length
    return: train, validation and test huggingface dataset object 
            and num_classes.
    """

    # loading yahoo answer dataset
    dataset = load_dataset('yahoo_answers_topics')

    # renaming columns for consistent interface
    dataset = dataset.rename_columns({"topic": "label",
                                      "question_title": "text"})

    # splitting train into train and validation
    train_val_split = dataset['train'].train_test_split(train_size=train_num,
                                                        load_from_cache_file=True,
                                                        shuffle=True,
                                                        seed=seed)

    return train_val_split['train'], train_val_split['test'], dataset['test']


class Dataloader(pl.LightningDataModule):

    def __init__(self, dataset, train_num, tok: str = "bert-base-cased", batch_size=32, num_workers=8, truncation=True,
                 max_length=256):
        super().__init__(Dataloader)
        self.train = None
        self.val = None
        self.test = None
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_num = train_num
        self.truncation = truncation
        self.max_length = max_length
        self.tokenizer = BertTokenizerFast.from_pretrained(tok)

    def setup(self) -> None:
        if self.dataset == 'YA':
            self.train, self.val, self.test = yahoo_answers_dataset(train_num=self.train_num)

        self.train = self.train.map(
            lambda e: self.tokenizer(e['text'], truncation=self.truncation, max_length=self.max_length,
                                     padding='max_length'), batched=True, load_from_cache_file=False)

        self.val = self.val.map(
            lambda e: self.tokenizer(e['text'], truncation=self.truncation, max_length=self.max_length,
                                     padding='max_length'), batched=True, load_from_cache_file=False)

        self.test = self.test.map(
            lambda e: self.tokenizer(e['text'], truncation=self.truncation, max_length=self.max_length,
                                     padding='max_length'), batched=True, load_from_cache_file=False)

        self.train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        self.val.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        self.test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)
