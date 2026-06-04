import io, sys, json
from difflib import SequenceMatcher
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, r'C:\Users\DELL\Desktop\CHM_extrait')

from query_docs import HybridRAG

with open(r'C:\Users\DELL\Desktop\CHM_extrait\test_questions.json', encoding='utf-8') as f:
    tests = {t['id']: t for t in json.load(f)['test_questions']}

rag = HybridRAG()
for tid in ['chemins_harmony_001', 'chemins_harmony_002', 'utilisateurs_001']:
    t = tests[tid]
    out = rag.query(t['question'], top_k=5)
    print('\n===', tid, '===')
    print('Q:', t['question'][:80])
    print('Expected:', t['expected_answer'][:100])
    for i, r in enumerate(out['merged_results'][:5]):
        src = r.get('source', '')[-60:]
        txt = (r.get('text') or '')[:200].replace('\n', ' ')
        exp_in = t['expected_answer'][:40].lower() in (r.get('text') or '').lower()
        print(f"  [{i}] {r.get('method')} score={r.get('rerank_score', r.get('hybrid_score',0)):.3f} exp_in={exp_in}")
        print(f"      src=...{src}")
        print(f"      txt={txt}")
