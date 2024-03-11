import math
import torch
import config
import collections
import numpy as np
from torch.optim.lr_scheduler import LambdaLR



class StatisticsReporter:
    def __init__(self):
        self.statistics = collections.defaultdict(list)

    def update_data(self, d):
        for k, v in d.items():
            if isinstance(v, (int, float)):
                self.statistics[k] += [v]

    def clear(self):
        self.statistics = collections.defaultdict(list)

    def to_string(self):
        string_list = []
        for k, v in sorted(list(self.statistics.items()), key=lambda x: x[0]):
            mean = np.mean(v)
            string_list.append('{}: {:.5g}'.format(k, mean))
        return ', '.join(string_list)

    def get_value(self, key):
        if key in self.statistics:
            value = np.mean(self.statistics[key])
            return value
        else:
            return None

    def items(self):
        for k, v in self.statistics.items():
            yield k, v


# save log data
def log_data(s):
    with open(config.OUTPUT_DIR + 'log.txt', 'a+', encoding='utf-8') as f:
        f.write(s + '\n')
    print(s)


# schedule learning rate
def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=1.0, last_epoch=-1
):
    '''
        Create a schedule with a learning rate that decreases following the
        values of the cosine function with several hard restarts, after a warmup
        period during which it increases linearly between 0 and 1.
    '''

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# generate new anime song lyrics
def gen_anime_lyrics(title: str, prompt_text: str, model, tokenizer=config.TOKENIZER,):
    if len(title)!= 0 or len(prompt_text)!= 0:
        prompt_text = '<s>' + title + '[CLS]' + prompt_text
        prompt_text = prompt_text.replace('\n', '\\n ')

        prompt_tokens = tokenizer.tokenize(prompt_text)
        prompt_token_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
        prompt_tensor = torch.LongTensor(prompt_token_ids)
        prompt_tensor = prompt_tensor.view(1, -1).to(config.DEVICE)

    else:
        prompt_tensor = None
    
    # model forward
    output_sequences = model.generate(
        input_ids=prompt_tensor,
        max_length=2048,
        # max_new_tokens=300,#1024
        top_p=0.95,
        top_k=40,
        temperature=0.5,
        do_sample=True,
        early_stopping=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=0,
        num_return_sequences=1
    )

    # convert model outputs to readable sentence
    generated_sequence = output_sequences.tolist()[0]
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_sequence)
    generated_text = tokenizer.convert_tokens_to_string(generated_tokens)
    generated_text = '\n'.join([s.strip() for s in generated_text.split('\\n')]).replace(' ', '\u3000').replace('<s>', '').replace('</s>', '\n\n---end---')
    
    title_and_lyric = generated_text.split('<SPC>', 1)

    if len(title_and_lyric) == 1:
        title, lyric = '' , title_and_lyric[0].strip()

    else:
        title, lyric = title_and_lyric[0].strip(), title_and_lyric[1].strip()

    return f'\t『{title}』\n\n{lyric}'