import os

path = "./midterm_3/lercio_headlines.txt"

options =\
"--model            gru   " +\
"--n_epochs         700    " +\
"--print_every      50   " +\
"--hidden_size      32    " +\
"--n_layers         4     " +\
"--learning_rate    0.01  " +\
"--chunk_len        100   " +\
"--batch_size       1000   " +\
"--cuda                   " 

command = f"python ./midterm_3/char-rnn-pytorch/train.py {path} {options}"


os.system(command)