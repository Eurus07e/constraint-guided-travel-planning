"""Generate public metric tables from the verified frozen manifest and analyses."""
import csv
import json
from pathlib import Path
from utils.paths import ROOT,FROZEN


def main():
    manifest=json.loads((FROZEN/'manifest.json').read_text())
    methods=manifest['methods'];assets=ROOT/'paper/source/assets'
    csv_path=ROOT/'results/metrics_summary.csv'
    names={r['method_id']:r['method'] for r in csv.DictReader(csv_path.open())}
    fields=['method','method_id','delivery_rate','commonsense_micro','commonsense_macro','hard_micro','hard_macro','final_pass_rate']
    with csv_path.open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=fields,lineterminator="\n");writer.writeheader()
        for method,spec in methods.items():writer.writerow({'method':names[method],'method_id':method,**spec['expected_scores']})
    main_ids=list(methods)[:9];columns=['final_pass_rate','commonsense_macro','hard_macro','commonsense_micro','hard_micro']
    top={col:sorted({methods[m]['expected_scores'][col] for m in main_ids},reverse=True) for col in columns}
    rows=[r'\begin{tabular}{@{}lrrrrr@{}}',r'\toprule',r'Method & Final Pass (\%) & Common. Macro (\%) & Hard Macro (\%) & Common. Micro (\%) & Hard Micro (\%) \\',r'\midrule']
    for method in main_ids:
        cells=[]
        for col in columns:
            value=methods[method]['expected_scores'][col];cell=f'{value:.2f}'
            if value==top[col][0]:cell=r'\textbf{'+cell+'}'
            elif len(top[col])>1 and value==top[col][1]:cell=r'\underline{'+cell+'}'
            cells.append(cell)
        label=names[method]+(r'$^{\dagger}$' if method in {'constraint_direct_json','generic_self_refine_r1'} else '')
        rows.append(label+' & '+' & '.join(cells)+r' \\')
    rows += [r'\bottomrule',r'\end{tabular}'];(assets/'results_table.tex').write_text('\n'.join(rows)+'\n')
    comparisons=json.loads((ROOT/'results/paired_analysis.json').read_text())['comparisons']
    labels=['Direct to selector','Selector to deterministic repair','Deterministic to seeded loop','Deterministic to CC-MAR','CC-MAR: remove critic','CC-MAR: remove board','CC-MAR: remove specialization']
    rows=[r'\begin{tabular}{@{}lrrrl@{}}',r'\toprule',r'Comparison & Gains & Losses & $\Delta$ (pp) & Paired 95\% CI (pp) \\',r'\midrule']
    for label,data in zip(labels,comparisons.values()):
        low,high=data['paired_bootstrap_95_ci_percentage_points'];rows.append(f"{label} & {data['gains']} & {data['losses']} & {data['delta_percentage_points']:+.2f} & [{low:+.2f}, {high:+.2f}]"+r' \\')
    (assets/'paired_update_table.tex').write_text('\n'.join(rows+[r'\bottomrule',r'\end{tabular}'])+'\n')
    audits=json.loads((ROOT/'results/audit_comparison.json').read_text())
    rows=[r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',r'Method & Legacy FP & Strict FP & Legacy FN & Strict FN \\',r'\midrule']
    for method,data in audits.items():
        before=data['versions']['legacy-v1']['confusion'];after=data['versions']['strict-v2']['confusion']
        rows.append(f"{names[method]} & {before['fp']} & {after['fp']} & {before['fn']} & {after['fn']}"+r' \\')
    (assets/'audit_update_table.tex').write_text('\n'.join(rows+[r'\bottomrule',r'\end{tabular}'])+'\n')
    # README headline rows are always sourced from the same frozen metric table.
    readme=ROOT/'README.md';text=readme.read_text();start=text.index('| Method | Final Pass |');end=text.index('\n\nThe complete table',start)
    lines=['| Method | Final Pass | Commonsense Macro | Hard Macro |','|---|---:|---:|---:|']
    for method in ['direct','generic_self_refine_r1','verifier_hybrid_agent','seeded_repair_probe','multi_agent_seeded_r2_n3','cc_mar_r3']:
        score=methods[method]['expected_scores'];lines.append(f"| {names[method]} | {score['final_pass_rate']:.2f} | {score['commonsense_macro']:.2f} | {score['hard_macro']:.2f} |")
    readme.write_text(text[:start]+'\n'.join(lines)+text[end:])
    print('Updated CSV, README headline table and three paper tables from frozen evidence.')


if __name__=='__main__':main()
