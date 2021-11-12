from datasets import load_dataset
def load_xsum_data():

    raw_datasets = load_dataset("xsum")
    #raw_datasets.shard(num_shards=20, index=0)
    #print(type(raw_datasets))
    return raw_datasets.filter(lambda example, indice: indice % 50 == 0, with_indices=True)
