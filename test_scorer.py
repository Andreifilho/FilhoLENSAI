from src.scorer import score_folder

results = score_folder('data/raw')

for r in results:
    print(r['score'], '—', r['filename'])