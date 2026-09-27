import os
import pandas as pd
import numpy as np

DATA_DIR = 'data'
ANSWER_FILE = 'answer.csv'
QUERIES_FILE = os.path.join(DATA_DIR, 'benchmark_queries.parquet')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.parquet')

def check_and_evaluate():
    if not os.path.exists(ANSWER_FILE):
        print(f"Error: {ANSWER_FILE} not found")
        return

    sub = pd.read_csv(ANSWER_FILE)
    queries = pd.read_parquet(QUERIES_FILE)
    
    print(f"Rows: {len(sub)} / Expected: {len(queries)}")

    assert len(sub) == len(queries), f"Row count mismatch: {len(sub)} vs {len(queries)}"
    assert list(sub.columns) == ['query_id', 'answer'], "Invalid columns. Required: 'query_id', 'answer'"

    bad_rows = 0
    dup_rows = 0
    for _, row in sub.iterrows():
        items = str(row['answer']).strip().split()
        if len(items) != 50:
            bad_rows += 1
        if len(set(items)) != len(items):
            dup_rows += 1

    if bad_rows:
        print(f"Warning: {bad_rows} rows don't have exactly 50 items")
    if dup_rows:
        print(f"Warning: {dup_rows} rows contain duplicates")
        
    if not bad_rows and not dup_rows:
        print("Format OK: 50 unique IDs per row")

    target_dict = {}
    if 'item_id' in queries.columns:
        target_dict = dict(zip(queries['query_id'].astype(str), queries['item_id'].astype(str)))
    elif os.path.exists(TRAIN_FILE):
        train = pd.read_parquet(TRAIN_FILE)
        if 'query_id' in train.columns and 'item_id' in train.columns:
            targets = train.groupby('query_id')['item_id'].last()
            target_dict = dict(zip(targets.index.astype(str), targets.values.astype(str)))

    if not target_dict:
        return

    mrr_scores, recall_scores = [], []
    for _, row in sub.iterrows():
        q_id = str(row['query_id'])
        if q_id in target_dict:
            target = target_dict[q_id]
            preds = str(row['answer']).strip().split()
            
            if target in preds:
                recall_scores.append(1.0)
                mrr_scores.append(1.0 / (preds.index(target) + 1))
            else:
                recall_scores.append(0.0)
                mrr_scores.append(0.0)

    if recall_scores:
        print(f"Evaluated: {len(recall_scores)}")
        print(f"MRR@50:    {np.mean(mrr_scores):.4f}")
        print(f"Recall@50: {np.mean(recall_scores):.4f}")
    else:
        print("No matching query IDs found in ground truth")

if __name__ == '__main__':
    check_and_evaluate()