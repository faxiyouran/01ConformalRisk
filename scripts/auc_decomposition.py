"""Mann–Whitney pair decomposition; paired task × label stratified bootstrap."""
from __future__ import annotations
from fractions import Fraction
import numpy as np
from common import ROOT,SPLITS,TASKS,REPLICATES,SEED,save_csv,rng_for,intervals


def credit2(positive,negative):
    """Twice the sum of 1 for wins and .5 for exact binary64 ties."""
    negative=np.sort(np.asarray(negative,dtype=float))
    left=np.searchsorted(negative,positive,side="left")
    right=np.searchsorted(negative,positive,side="right")
    return int(np.sum(left+right,dtype=np.int64))


def decompose(scores,harm,tasks):
    scores=np.asarray(scores);harm=np.asarray(harm);tasks=np.asarray(tasks)
    npos=int(harm.sum());nneg=len(harm)-npos;total=npos*nneg
    assert npos>0 and nneg>0
    total2=credit2(scores[harm==1],scores[harm==0]);within=0;within2=0
    for task in TASKS:
        p=scores[(tasks==task)&(harm==1)];n=scores[(tasks==task)&(harm==0)]
        within+=len(p)*len(n);within2+=credit2(p,n)
    between=total-within;between2=total2-within2
    assert within>0 and between>0
    # Integer pair accounting and exact rational identity, including ties.
    assert Fraction(total2,2*total)==Fraction(within,total)*Fraction(within2,2*within)+Fraction(between,total)*Fraction(between2,2*between)
    a=total2/(2*total);w=within2/(2*within);b=between2/(2*between)
    return dict(n_positive=npos,n_negative=nneg,total_pairs=total,total_pair_credit_twice=total2,
        within_pairs=within,between_pairs=between,within_pair_credit_twice=within2,between_pair_credit_twice=between2,
        weight_within=within/total,weight_between=between/total,auc_overall=a,auc_within=w,auc_between=b,
        contribution_within=within2/(2*total),contribution_between=between2/(2*total),between_minus_within=b-w)


def task_pairs(df,split):
    out=[];allpairs=int(df.harm_indicator.sum())*int((1-df.harm_indicator).sum())
    for positive_task in TASKS:
        p=df[(df.task_type==positive_task)&(df.harm_indicator==1)].guardian_score.to_numpy()
        for negative_task in TASKS:
            n=df[(df.task_type==negative_task)&(df.harm_indicator==0)].guardian_score.to_numpy()
            pairs=len(p)*len(n);twice=credit2(p,n)
            out.append(dict(split=split,positive_task=positive_task,negative_task=negative_task,
                relation="within" if positive_task==negative_task else "between",n_positive=len(p),n_negative=len(n),pairs=pairs,
                credit_twice=twice,auc=twice/(2*pairs) if pairs else None,pair_share=pairs/allpairs,
                weighted_contribution=twice/(2*allpairs),status="DEFINED" if pairs else "UNDEFINED_NO_POSITIVE_OR_NEGATIVE_SUPPORT"))
    return out


def run(df,root=ROOT):
    points=[];cis=[];cells=[]
    for split_index,split in enumerate(SPLITS):
        panel=df[df.split==split].sort_values("episode_id").reset_index(drop=True)
        scores=panel.guardian_score.to_numpy();harm=panel.harm_indicator.to_numpy();tasks=panel.task_type.to_numpy()
        point=decompose(scores,harm,tasks);points.append(dict(split=split,N=len(panel),**point))
        cells.extend(task_pairs(panel,split))
        strata=[np.flatnonzero((tasks==t)&(harm==h)) for t in TASKS for h in (0,1)]
        strata=[ix for ix in strata if len(ix)]
        rng=rng_for(1,split_index);boot={key:[] for key in ("auc_overall","auc_within","auc_between","between_minus_within")}
        for _ in range(REPLICATES):
            draw=np.concatenate([rng.choice(ix,size=len(ix),replace=True) for ix in strata])
            d=decompose(scores[draw],harm[draw],tasks[draw])
            assert (d["total_pairs"],d["within_pairs"],d["between_pairs"])==(point["total_pairs"],point["within_pairs"],point["between_pairs"])
            for key in boot:boot[key].append(d[key])
        for key,v in boot.items():
            low,high=intervals(v)
            cis.append(dict(split=split,metric=key,estimate=point[key],ci_lower=low,ci_upper=high,
                replicates=REPLICATES,seed=SEED,rng_stream=f"SeedSequence([{SEED},1,{split_index},0])",strata="task_type x harm_indicator",
                method="paired stratified episode percentile bootstrap; linear quantiles",scope="retrospective diagnostic analysis; conditional on stratum counts and fixed scores"))
    save_csv(root/"results/auc_decomposition.csv",points);save_csv(root/"results/auc_bootstrap_intervals.csv",cis)
    save_csv(root/"results/auc_task_pair_components.csv",cells)
    return points,cis,cells
