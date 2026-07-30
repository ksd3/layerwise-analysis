import json
with open('metrics/manifold_emergence.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
print('Notebook cells:', len(nb['cells']))
for i, c in enumerate(nb['cells']):
    cid = c.get('id', c.get('cell_id', '?'))
    print(f"{i:2d} {c['cell_type']:9s} {cid}")