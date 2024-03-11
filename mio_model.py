import torch
import config
from torch.nn import functional as F
from transformers import AutoModelForCausalLM



model = AutoModelForCausalLM.from_pretrained(
    config.CHECKPOINT,
    device_map='auto',
    token=config.ACCESS_TOKEN,
)


def forward_step(batch_data, model, tokenizer=config.TOKENIZER,):
    # Get max length
    max_len = max([len(sequins) for sequins in batch_data])

    # Padding input sequences
    batch_data = [sequins + [0] * (max_len - len(sequins)) for sequins in batch_data]

    # Convert to tensors
    batch_tensor = torch.LongTensor(batch_data).to(model.device)

    # Get inputs and outputs
    input_ids = batch_tensor[:, :-1].contiguous()
    output_ids = batch_tensor[:, 1:].contiguous()

    # Forward
    karasu_outputs = model(input_ids=input_ids, return_dict=True, )

    loss = F.cross_entropy(
        input=karasu_outputs['logits'].view(-1, tokenizer.vocab_size),
        target=output_ids.view(-1),
        ignore_index=0,
        reduction='mean'
    )

    with torch.no_grad():
        ppl = loss.exp()

    return loss, ppl

