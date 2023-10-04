# Importing the libraries
import torch.nn as nn
import torch.nn.functional as F
import torch
from model import TransformerModel
import json
import argparse

# Loading the model config
with open('model//config.json', 'r') as f:
    config = json.load(f)

# Creating the stoi and itos dictionaries
stoi = config['stoi']
itos = config['itos']

# Creating a lambda function to convert a string to a integer
encode = lambda s: [stoi[c] for c in s]

# Creating a lambda function to convert a integer to a string
decode = lambda i: ''.join([itos[str(j)] for j in i])

# Creating the generate function
def generate(context:str, model:nn.Module, max_tokens:int=100, temperature:int=1, 
             block_size:int=32, device:str='cpu', seed:int=None) -> str:
    # Setting the model to evaluation mode
    model.eval()

   # Context string
    context = context

    # Print the context
    print(context, end='')

    # Convert the context to tokens
    context_tokens = torch.Tensor([encode(context[-block_size:])]).long().to(device)

    # Set the random seed
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    for _ in range(max_tokens):
        # Generate the next token
        generated = model(context_tokens) * temperature
        
        # Sample a token from the probability distribution
        next_token = torch.multinomial(torch.softmax(generated[:, -1, :], dim=-1), 
                                       num_samples=1, generator=generator).squeeze()

        # Decode the generated token
        decoded = decode([next_token.item()])[0]
        print(decoded, end='')

        # Append the generated token to the context
        context += decoded

        if context_tokens.shape[1] >= block_size:
            # Remove the first token
            context_tokens = context_tokens[:, 1:]

        # Concatenate the generated token and the context tokens
        context_tokens = torch.cat([context_tokens, next_token.view(1, 1)], dim=-1)

    return context

def main():
    # Creating the argument parser
    parser = argparse.ArgumentParser(description='Generate text using the Transformer model')

    # Adding the arguments
    parser.add_argument('--context', type=str, help='The context to start the generation from')
    parser.add_argument('--max_tokens', type=int, help='The maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, help='The temperature to use for the generation', default=1.0)
    parser.add_argument('--seed', type=int, help='The random seed to use for the generation', default=None)
    parser.add_argument('--device', type=str, help='The device to use for the generation', default='cpu')

    # Parsing the arguments
    args = parser.parse_args()

    # Get the context, max tokens, temperature, and seed
    context = args.context
    max_tokens = args.max_tokens
    temperature = args.temperature
    seed = args.seed

    # Hyperparameters and settings
    vocab_size = config['vocab_size']
    n_embed = config['n_embed']
    n_heads = config['n_heads']
    block_size = config['block_size']
    n_layers = config['n_layers']
    dropout_rate = config['dropout_rate']

    # Creating the model
    transformer = TransformerModel(vocab_size, n_embed, n_heads, block_size, n_layers, dropout_rate)

    # Loading the model in
    transformer.load_state_dict(torch.load('model//model_weights.pth'))

    # Setting the device
    device = torch.device(args.device)

    # Moving the model to the device
    transformer.to(device)

    # Generate the text
    generate(context, transformer, max_tokens, temperature, block_size, device, seed)

    # Print a new line
    print()

if __name__ == "__main__":
    main()
