import unicodedata
import numpy
from tqdm import tqdm
import config


def get_token_ids(text, tokenizer):

    text = text.strip()
    text = text.replace('\n', '\\n')
    text = unicodedata.normalize('NFKC', text)
    text = ''.join(i for i in text if i.isprintable())
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return token_ids


def get_docs(lyrics_dataset):

    docs = []
    for song in tqdm(lyrics_dataset):
        ids = get_token_ids(f'{song["SongTitle"]}<SPC>{song["Lyric"]}', config.TOKENIZER)
        if config.TOKENIZER.unk_token_id not in ids:
            docs.append(ids)

    print(f'docs length: {len(docs)}')    
    print(f'docs mean: {numpy.mean([len(i) for i in docs])}')

    return docs


