import json, glob
def g(d, *ks):
    for k in ks:
        d = (d or {}).get(k) if isinstance(d, dict) else None
    return d
print("== Track B ES (ColVision) ==")
for f in sorted(glob.glob('/mloscratch/users/mbonnet/bcv-dev/results/track_b/*/es/seed_0.json')):
    d = json.load(open(f))
    r = d.get('retrieval') or {}
    print('NDCG', f.split('/')[-3], 'ndcg@5=', r.get('ndcg_at_5'), '| ndcg@10=', r.get('ndcg_at_10'), '| recall@5=', r.get('recall_at_5'), '| notes=', (d.get('notes') or '')[:40])
print("== Baseline texte ES ==")
for f in sorted(glob.glob('/mloscratch/users/mbonnet/bcv-dev/results/baseline_text/B/es/seed_0.json')):
    d = json.load(open(f))
    r = d.get('retrieval') or {}
    print('BASE', 'ndcg@5=', r.get('ndcg_at_5'), '| ndcg@10=', r.get('ndcg_at_10'))
print("AGG_DONE")
