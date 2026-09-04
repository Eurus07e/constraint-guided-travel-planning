"""Paired, reproducible uncertainty estimates for frozen benchmark predictions."""
import json
import math
from pathlib import Path
import numpy as np
from utils.paths import FROZEN,ROOT


def paired_stats(before,after,bootstrap_samples=20000,seed=20260904):
    if len(before)!=len(after) or not before:raise ValueError('Paired nonempty aligned outcomes required')
    if any(type(x) is not bool for x in list(before)+list(after)):raise ValueError('Outcomes must be booleans')
    delta=np.asarray(after,dtype=int)-np.asarray(before,dtype=int)
    gains=int((delta==1).sum());losses=int((delta==-1).sum());discordant=gains+losses
    p=min(1,2*sum(math.comb(discordant,k) for k in range(min(gains,losses)+1))/2**discordant) if discordant else 1.0
    rng=np.random.default_rng(seed)
    # Multinomial resampling is exactly the empirical paired bootstrap of deltas.
    counts=rng.multinomial(len(delta),[losses/len(delta),1-discordant/len(delta),gains/len(delta)],size=bootstrap_samples)
    intervals=np.quantile((counts[:,2]-counts[:,0])/len(delta),[0.025,0.975])
    return {'n':len(delta),'before_pass':sum(before),'after_pass':sum(after),'gains':gains,'losses':losses,
        'delta_percentage_points':100*float(delta.mean()),'paired_bootstrap_95_ci_percentage_points':[100*float(x) for x in intervals],
        'mcnemar_exact_two_sided_p':float(p),'bootstrap_samples':bootstrap_samples,'bootstrap_seed':seed}


def main():
    manifest=json.loads((FROZEN/'manifest.json').read_text());methods=manifest['methods']
    comparisons=[('direct','verifier_hybrid_agent'),('verifier_hybrid_agent','seeded_repair_probe'),('seeded_repair_probe','multi_agent_seeded_r2_n3'),('seeded_repair_probe','cc_mar_r3'),('cc_mar_r3','cc_mar_no_critic'),('cc_mar_r3','cc_mar_no_board'),('cc_mar_r3','cc_mar_no_specialization')]
    result={'scope':'Conditional on these frozen predictions and the development split; intervals do not measure model rerun uncertainty. Component contrasts are exploratory, not isolated causal effects.',
        'comparisons':{a+'_to_'+b:paired_stats(methods[a]['final_pass'],methods[b]['final_pass']) for a,b in comparisons}}
    (ROOT/'results/paired_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
