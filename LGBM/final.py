import json

path = 'data/IND-WhoIsWho/sub/'
with open(path + 'result-text-20240519-095242-3e-5-2.json', 'r') as f:  # llma3
    result_llm = json.load(f)
with open(path + 'result-text-20240601-141542-3e-4.json', 'r') as f:  # chatglm
    result_llm2 = json.load(f)
with open(path + 'testb_baseline_795.json', 'r') as f:  # lgb
    result_lgb = json.load(f)
with open(path + 'gcn.json', 'r') as f:  # gcn
    result_gcn = json.load(f)

for id, names in submission.items():
    for name in names:
        submission[id][name] = result_llm[id][name] * 0.4 + result_llm2[id][name] * 0.3 + result_lgb[id][name] * 0.2 + \
                               result_gcn[id][name] * 0.1

with open(path + 'test_b_lgb_llm2_gcn_final.json', 'w', encoding='utf-8') as f:
    json.dump(submission, f, ensure_ascii=False, indent=4)