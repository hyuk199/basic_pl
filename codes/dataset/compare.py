from torch.utils.data import Dataset, DataLoader
import torch
import pytorch_lightning as pl
from rich.progress import track

import pandas as pd

def create_diff_dataset(path):
    df = pd.read_csv(f"{path}",
                    dtype= {'id': str,'text': str, 'difficulty':float}).set_index('id')

    df1 = df.loc[df['difficulty'] > 0]
    df2 = df.loc[df['difficulty'] <= 0]

    pairs = []
    # chosen must bigger
    for rejected_summary, rejected_score, chosen_summary, chosen_score in zip(df1['text'],df1['difficulty'], df2['text'],df2['difficulty']):
        pair = {}
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        pair["chosen"] = chosen_summary
        pair["chosen_score"] = chosen_score
        pair["rejected"] = rejected_summary
        pair["rejected_score"] = rejected_score
        pairs.append(pair)
    
    return pairs

def create_diff_over_dataset(path):
    df = pd.read_csv(f"{path}",
                    dtype= {'id': str,'text': str, 'difficulty':float}).set_index('id')
    
    difficulty_list = df.groupby('difficulty').count().index.values
    pairs = []
    for diff in difficulty_list:

        df1 = df.loc[df['difficulty'] > diff]
        df2 = df.loc[df['difficulty'] == diff]
        # chosen must bigger
        for rejected_summary, rejected_score, chosen_summary, chosen_score in zip(df1['text'],df1['difficulty'], df2['text'],df2['difficulty']):
            pair = {}
            if chosen_summary == rejected_summary:
                continue
            if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
                continue
            pair["chosen"] = chosen_summary
            pair["chosen_score"] = chosen_score
            pair["rejected"] = rejected_summary
            pair["rejected_score"] = rejected_score
            pairs.append(pair)
    
    return pairs

def create_diff_upper_dataset(path):
    df = pd.read_csv(f"{path}",
                    dtype= {'id': str,'text': str, 'difficulty':float}).set_index('id')

    difficulty_list = df.groupby('difficulty').count().index.values
    pairs = []
    old_diff = difficulty_list[0]
    for diff in difficulty_list[1:]:

        df1 = df.loc[df['difficulty'] == diff]
        df2 = df.loc[df['difficulty'] == old_diff]
        # chosen must bigger
        for rejected_summary, rejected_score, chosen_summary, chosen_score in zip(df1['text'],df1['difficulty'], df2['text'],df2['difficulty']):
            pair = {}
            if chosen_summary == rejected_summary:
                continue
            if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
                continue
            pair["chosen"] = chosen_summary
            pair["chosen_score"] = chosen_score
            pair["rejected"] = rejected_summary
            pair["rejected_score"] = rejected_score
            pairs.append(pair)
        old_diff = diff
    
    return pairs

def create_unknown_dataset(path):
    df = pd.read_csv(f"{path}",
                    dtype= {'id': str,'text': str, 'unknown':bool}).set_index('id')
    
    df1 = df.loc[df['unknown'] == True]
    df1 = df1.sample(frac=1) 
    df2 = df.loc[df['unknown'] == False]
    

    pairs = []
    # chosen must bigger
    for rejected_summary, chosen_summary in zip(df1['text'], df2['text']):
        pair = {}
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        pair["chosen"] = chosen_summary
        pair["rejected"] = rejected_summary
        pairs.append(pair)
    
    return pairs

class CompareDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        enumerator = enumerate(track(pairs, f"Loading Dataset"))
        for i, pair in enumerator:
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer(
                chosen,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                rejected,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
            self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
            self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
            self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )

def collate_datastruct_and_text(data):
    batch = {}
    batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
    batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
    batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
    return batch

class Datamodule(pl.LightningDataModule):
    def __init__(self, 
                data_path: str,
                batch_size: int,
                num_workers: int,
                shuffle = True,
                **kwargs
                ):
        """ Basic datamodule
            Need to set 
                __init__()
                prepare_data() : load data to data_path to prepare data
                setup() : set dataset to train, validation, test dataset
        """

        super().__init__()
        
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def prepare_data(self) -> None:
        pass
    
    def setup(self, stage='fit'):
        # -- dataset building
        if stage == "fit":
            self.train_dataset = CompareDataset(
                self.data_path
            )
            self.valid_dataset = CompareDataset(
                self.data_path
            )

        if stage == "test":
            self.test_dataset = CompareDataset(
                self.data_path
            )
        
    def train_dataloader(self):
        return DataLoader(
                    self.train_dataset, 
                    batch_size = self.batch_size, 
                    num_workers = self.num_workers,
                    shuffle=self.shuffle,
                    collate_fn= collate_datastruct_and_text
                )

    def val_dataloader(self):
        return DataLoader(
                    self.valid_dataset, 
                    batch_size = self.batch_size, 
                    num_workers = self.num_workers,
                    shuffle=False,
                    collate_fn= collate_datastruct_and_text
                )


    def test_dataloader(self):
        return DataLoader(
                    self.test_dataset, 
                    batch_size = self.batch_size, 
                    num_workers = self.num_workers,
                    shuffle=False,
                    collate_fn= collate_datastruct_and_text
                )
