from src.scorer import score_folder

results = score_folder('data/raw/aum')

for r in results:
    print(r['score'], '—', r['filename'])