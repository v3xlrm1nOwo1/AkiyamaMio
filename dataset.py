import torch
import config
from torch.utils.data import RandomSampler


class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, lyrics_tachi_tokens_ids, stage, max_sequence_length=config.MAX_SEQUENCE_LENGTH, tokenizer=config.TOKENIZER,):
        
        # Attributes
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.stage = stage

        # Load dataset
        self.lyrics = lyrics_tachi_tokens_ids

        # Calculate basic statistics
        self.statistics = {'num_lyrics_tachi_tokens_ids': len(self.lyrics)}

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, idx):
        lyrics = self.lyrics[idx]
        lyrics = [self.tokenizer.bos_token_id] + lyrics + [self.tokenizer.eos_token_id]

        lyrics = lyrics[:self.max_sequence_length]

        return lyrics


def collate_fn(batch_data):
    return batch_data


def create_data_loader(batch_size,
                       stage, 
                       lyrics_tachi_tokens_ids, 
                       num_workers=config.NUM_WORKERS, 
                       collate_fn=collate_fn, 
                       max_len=config.MAX_SEQUENCE_LENGTH, 
                       tokenizer=config.TOKENIZER,):
    
    
    ds = LyricsDataset(
        max_sequence_length=max_len,
        lyrics_tachi_tokens_ids=lyrics_tachi_tokens_ids,
        tokenizer=tokenizer,
        stage=stage,
        )

    print(f'length docs: {str(ds.statistics)}')

    return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            sampler=RandomSampler(ds, replacement=False),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            # drop_last=True,
        )

