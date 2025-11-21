import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
results_file = os.path.join(script_dir, 'hybrid_retrieval_semantic_results.json')

with open(results_file, 'r') as f:
    data = json.load(f)

fn_cases = [r for r in data['results'] if r['should_be_relevant'] and not r['evaluation']['is_relevant']]

print(f'False Negatives: {len(fn_cases)}\n')
for i, r in enumerate(fn_cases, 1):
    print(f"{i}. [{r['semantic_category']}] {r['query']}")
    print(f"   Retrieved: {r['retrieved_docs']} docs, Relevance: {r['evaluation']['overall_relevance']:.3f}")
    print(f"   Topic relevance: {r['evaluation']['topic_relevance']:.3f}, Keyword: {r['evaluation']['keyword_relevance']:.3f}")
    print()

