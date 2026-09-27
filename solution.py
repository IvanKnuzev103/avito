import os
import gc
import numpy as np
import pandas as pd
import Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DATA_DIR = 'data'

train_df = pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet'))
queries_df = pd.read_parquet(os.path.join(DATA_DIR, 'benchmark_queries.parquet'))
items_df = pd.read_parquet(os.path.join(DATA_DIR, 'benchmark_items.parquet'))

items_text = (
    items_df['item_title_raw'].fillna('') + ' ' + 
    items_df['item_infm_params_text'].fillna('')
).str.lower()

queries_text = (
    queries_df['search_query'].fillna('') + ' ' +
    queries_df['search_infm_params_text'].fillna('')
).str.lower()

has_categories = 'microcat_id' in items_df.columns and 'microcat_id' in queries_df.columns
if has_categories:
    item_cats = items_df['microcat_id'].values
    query_cats = queries_df['microcat_id'].values
elif 'category_id' in items_df.columns and 'category_id' in queries_df.columns:
    item_cats = items_df['category_id'].values
    query_cats = queries_df['category_id'].values
    has_categories = True

del items_df, queries_df
gc.collect()

en_stemmer = Stemmer.Stemmer('english')
ru_stemmer = Stemmer.Stemmer('russian')

def stemmed_words(doc):
    analyzer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b').build_analyzer()
    for token in analyzer(doc):
        yield ru_stemmer.stemWord(en_stemmer.stemWord(token))

word_vec = TfidfVectorizer(
    analyzer=stemmed_words,
    max_features=60000,
    sublinear_tf=True
)

char_vec = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 5),
    max_features=60000,
    sublinear_tf=True
)

X_items_w = normalize(word_vec.fit_transform(items_text), axis=1)
X_queries_w = normalize(word_vec.transform(queries_text), axis=1)

X_items_c = normalize(char_vec.fit_transform(items_text), axis=1)
X_queries_c = normalize(char_vec.transform(queries_text), axis=1)

del items_text, queries_text, word_vec, char_vec
gc.collect()

items_df_ids = pd.read_parquet(os.path.join(DATA_DIR, 'benchmark_items.parquet'), columns=['item_id'])
queries_df_ids = pd.read_parquet(os.path.join(DATA_DIR, 'benchmark_queries.parquet'), columns=['query_id'])

item_ids = items_df_ids['item_id'].astype(str).values
query_ids = queries_df_ids['query_id'].astype(str).values

popular_items = train_df['item_id'].value_counts().index.astype(str).tolist()
fallback_items = popular_items[:50] if len(popular_items) >= 50 else list(item_ids[:50])

del train_df, items_df_ids, queries_df_ids
gc.collect()

predictions = []
batch_size = 100
num_queries = X_queries_w.shape[0]

for i in range(0, num_queries, batch_size):
    qw = X_queries_w[i:i + batch_size]
    qc = X_queries_c[i:i + batch_size]
    
    scores = (qw.dot(X_items_w.T).toarray() * 0.5) + (qc.dot(X_items_c.T).toarray() * 0.5)
    
    if has_categories:
        batch_q_cats = query_cats[i:i + batch_size]
        for q_idx, q_cat in enumerate(batch_q_cats):
            if pd.notna(q_cat):
                scores[q_idx, item_cats == q_cat] *= 1.4

    for row in scores:
        top_indices = np.argpartition(row, -50)[-50:]
        top_indices = top_indices[np.argsort(-row[top_indices])]
        
        top_ids = item_ids[top_indices].tolist()
        
        if len(top_ids) < 50 or row[top_indices[0]] == 0:
            existing = set(top_ids)
            for fb in fallback_items:
                if fb not in existing:
                    top_ids.append(fb)
                if len(top_ids) == 50:
                    break
                    
        predictions.append(' '.join(top_ids[:50]))

pd.DataFrame({
    'query_id': query_ids,
    'answer': predictions
}).to_csv('answer.csv', index=False)
