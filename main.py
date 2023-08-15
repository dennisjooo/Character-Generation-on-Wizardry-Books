# Importing the libraries
import torch.nn as nn
import torch.nn.functional as F
import torch
from model import TransformerModel
import joblib
import sys

# Loading the stoi and itos
maps = joblib.load('model//stoi_itos.pkl')
stoi = maps['stoi']
itos = maps['itos']

# Creating a lambda function to convert a string to a integer
encode = lambda s: [stoi[c] for c in s]

# Creating a lambda function to convert a integer to a string
decode = lambda i: ''.join([itos[j] for j in i])

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
    # Check if the number of arguments is correct
    if len(sys.argv) < 4:
        print("Usage: python generate.py <context> <max_tokens> <temperature> <seed(optional)>")
        return

    # Get the context, max tokens, temperature, and seed
    context = sys.argv[1]
    max_tokens = int(sys.argv[2])
    temperature = float(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) == 5 else None

    # Hyperparameters and settings
    vocab_size = 92
    n_embed = 512
    n_heads = 16
    block_size = 32
    n_layers = 8
    dropout_rate = 0.1

    # Creating the model
    transformer = TransformerModel(vocab_size, n_embed, n_heads, block_size, n_layers, dropout_rate)

    # Loading the model in
    transformer.load_state_dict(torch.load('model//model_weights.pth'))

    # Setting the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Moving the model to the device
    transformer.to(device)

    # Generate the text
    generate(context, transformer, max_tokens, temperature, block_size, device, seed)

if __name__ == "__main__":
    main()
