# autoregressive generation

import torch

def generate_text(model, input_ids, max_new_tokens, block_size):
    model.eval()

    generated_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = generated_ids[:, -block_size:]
            logits = model(context)
            
            next_token_logits = logits[:, -1, :]
            
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

    # print("context:", context.shape)
    # print("logits:", logits.shape)
    # print("next_token_logits:", next_token_logits.shape)
    # print("next_token_id:", next_token_id.shape)

    return generated_ids