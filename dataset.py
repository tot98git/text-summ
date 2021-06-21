import tensorflow_datasets as tfds
import nltk

# extract cnn dailymail documents to text files using the tensfor flow data processors


def get_data():
    ds = tfds.load('cnn_dailymail', split='test', shuffle_files=True)
    df = tfds.as_dataframe(ds)

    lengths = []
    sum_lengths = []
    compression = []

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    for idx in range(0, 5000):
        article = str(df.iloc[idx]['article'])
        highlights = str(df.iloc[idx]['highlights']).replace("\\n", " ")

        article_len = len(tokenizer.tokenize(article[1:]))
        summ_len = len(tokenizer.tokenize(highlights[1:]))

        lengths.append(article_len)
        sum_lengths.append(summ_len)
        compression.append(summ_len/article_len)

        file = open(f"cnn_daily/{idx}_article.txt", "w")
        file.write(str(article)[1:])
        file.close()

        file = open(f"cnn_daily/{idx}_summ.txt", "w")
        file.write(str(highlights)[1:])
        file.close()

    print(sum(lengths)/len(lengths), sum(sum_lengths) /
          len(lengths), sum(compression)/len(lengths))


get_data()
