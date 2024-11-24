def load_dataset(filename):
    import pickle
    with open(filename, 'rb') as handle:
        dic = pickle.load(handle)
    return dic

def run(filename):
    f = open('dd.txt','w')
    dic = load_dataset(filename)
    for prompt, res in dic['dataset']:
        f.write(prompt + '\n\n\n')
        response_txt = res['response']
        f.write(response_txt + '\n\n\n')
    f.close()

    

