import numpy as np
import matplotlib.pyplot as plt

from generate import generate
import sys


real_words = {}
with open('./dataset/parole.txt', 'r') as f:
    for p in f:
        real_words[p.strip().lower()] = True


def percetage_correct_words(real_words, model_path, temperature, num_titles, len_titles):
    total_words = 0
    wrong_words = 0
    for i in range(num_titles):
        title = generate(model_path, temperature, len_titles)
        # remove the ^ character
        title = title[1:]
        # remove the ~ characters
        title = title.replace('~', '')
        # for each word in the title
        for word in title.split():
            # if the word is not in the dictionary
            if word.lower() not in real_words:
                wrong_words += 1
            total_words += 1
    
    return 1 - wrong_words/total_words          


hidden_sizes = [100, 200, 300, 400]
layers = [1, 2, 3, 4]
temperatures = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

results = np.ndarray((len(hidden_sizes), len(layers), len(temperatures)), dtype=float)
for k , hidden_size in enumerate(hidden_sizes):
    for i, layer in enumerate(layers):
        for j, temperature in enumerate(temperatures):
            try:
                print(f'hidden_size: {hidden_size}, layer: {layer}, temperature: {temperature}', file=sys.stderr)
                results[k][i][j] = percetage_correct_words(
                    real_words=real_words, 
                    model_path=f'./models/lercio_E6000_H{hidden_size}_L{layer}.pt', 
                    temperature=temperature, 
                    num_titles=100, 
                    len_titles=200
                )
            except:
                results[k][i][j] = 0.0
                print('Error', file=sys.stderr)

# save the results
np.save('./results.npy', results)
