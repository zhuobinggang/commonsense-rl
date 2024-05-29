from datetime import datetime

def load_dataset(filename):
    import pickle
    with open(filename, 'rb') as handle:
        dic = pickle.load(handle)
    return dic

def get_time_str():
    # get time now
    dt = datetime.now()
    # format it to a string
    return dt.strftime('%Y%m%d_%H%M%S')
