# Extractive approach for text summarization using graphs

Natural language processing is an important discipline with the aim of understanding text by its digital representation, that due to the diverse way we write and speak, is often not accurate enough. Our paper explores different graph related algorithms that can be used in solving the text summarization problem using an extractive approach. We consider two metrics: sentence overlap and edit distance for measuring sentence similiarity. 

Our contribution includes a direct implementation of graph-based algorithms for computing relevant data that could summarize the test documents the best. The respective algorithms help us extract the most important sentences that will constitute the summary. We test two different metrics for computing sentence similiarity, and in one of them we employ a notion of graph similiarity - edit distance.

| File       | Description                                                                                                                                           |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| main.py    | Main file where the graph related structures have been setup.                                                                                         |
| data.py    | File that pre-processes the data - tokenization, stemming, ngram generators.                                                                          |
| dataset.py | File that uses tensorflow to convert the dataset of our choice to separate txt files for each article and summary (index_article.txt, index_summ.txt) |

The respective paper can be accessed here: https://arxiv.org/abs/2106.10955.
