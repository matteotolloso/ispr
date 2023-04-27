# https://github.com/spro/char-rnn.pytorch

import torch
from helpers import *
from model import *

def _generate(decoder, prime_str, predict_len, temperature, cuda):
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    return predicted

# Run as standalone script
def generate(filename : str, temperature : float, predict_len : int = 168):

    prime_str = '^'
    cuda = True

    decoder = torch.load(filename)
    return _generate(decoder, prime_str=prime_str, predict_len=predict_len, temperature=temperature, cuda = cuda )

if __name__ == '__main__':
    generate('lercio_padded.pt', 0.8)