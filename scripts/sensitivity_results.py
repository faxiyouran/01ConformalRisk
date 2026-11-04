"""Replay archived intervals and sensitivity analyses; never generate proposals or fit."""
from pathlib import Path
from fractions import Fraction as F
import csv,json,hashlib,math
import numpy as np
from scipy.special import expit
import historical_statistics as hs
from common import save_csv
from auc_decomposition import credit2

def read(path):
    with Path(path).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))

def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')

def auc_value(harm,score):
    p=score[harm==1];n=score[harm==0]
    return credit2(p,n)/(2*len(p)*len(n)) if len(p) and len(n) else None

def run(root,out):
    root=Path(root);out=Path(out)/'sensitivity';out.mkdir()
    config=json.loads((root/'data/sensitivity_config.json').read_text(encoding='utf8'))
    ep=read(root/'data/episodes.csv');policy=read(root/'data/policy_decisions.csv')
    permits={(r['episode_id'],r['policy']):int(r['execute']) for r in policy}
    byid={r['episode_id']:r for r in ep};assert len(byid)==4000
    cohorts={};primary=[];primary_provenance=[]
    for split,name in [('TEST','TEST'),('INDEPENDENT_CONFIRMATION','CONFIRMATION')]:
        rr=sorted([r for r in ep if r['split']==split],key=lambda r:r['episode_id'])
        c=dict(ids=[r['episode_id'] for r in rr],harm=np.array([int(r['harm_indicator']) for r in rr]),score=np.array([float(r['guardian_score']) for r in rr]),ieu=np.array([float(F(r['original_IEU_G0'])) for r in rr]),payment=np.array([float(F(r['payment_completion'])) if r['payment_completion'] else 0. for r in rr]),mask=np.array([r['task_type']=='OUTBOUND_PAYMENT' for r in rr]))
        cohorts[name]=c
        executions=np.array([[permits[r['episode_id'],p] for p in ('G0','G1','G2','G3')] for r in rr])
        protocol=config['primary_bootstrap'][name]
        result=hs.bootstrap(c['harm'],executions,c['ieu'],c['payment'],c['mask'],protocol)
        primary.extend(dict(cohort=name,**r,replicates=result['replicates'],seed=result['seed'],scope='FROZEN_PRIMARY_PAIRED_RESAMPLING') for r in result['rows'])
        primary_provenance.append(dict(cohort=name,replicates=result['replicates'],seed=result['seed'],indices_sha256=result['indices_sha256']))
    save_csv(out/'primary_intervals.csv',primary)
    save_json(out/'primary_bootstrap_provenance.json',primary_provenance)
    print('PASS: original 10,000-draw primary paired confidence intervals.',flush=True)
    # Same historical task order and one continuing RNG stream; negative then positive.
    taskcfg=config['task_auc_bootstrap'];assert taskcfg['replicates']==2000 and taskcfg['seed']==20261102
    assert taskcfg['task_order']==['DAPP_APPROVAL','PORTFOLIO_REBALANCE'] and taskcfg['strata_order']==[0,1]
    rng=np.random.default_rng(taskcfg['seed']);taskrows=[]
    for task in taskcfg['task_order']:
        # Original retained row order, not a new sort of the bootstrap population.
        rr=[r for r in ep if r['split']=='INDEPENDENT_CONFIRMATION' and r['task_type']==task]
        harm=np.array([int(r['harm_indicator']) for r in rr]);score=np.array([float(r['guardian_score']) for r in rr])
        negative=np.where(harm==0)[0];positive=np.where(harm==1)[0];boot=[]
        for _ in range(taskcfg['replicates']):
            draw=np.r_[rng.choice(negative,len(negative),replace=True),rng.choice(positive,len(positive),replace=True)]
            boot.append(auc_value(harm[draw],score[draw]))
        low,high=np.quantile(boot,[.025,.975],method='linear')
        taskrows.append(dict(task=task,n=len(rr),harm_cases=int(harm.sum()),auc=auc_value(harm,score),ci_low=float(low),ci_high=float(high),replicates=2000,seed=20261102,scope='RETROSPECTIVE_TASK_STRATIFIED_RESAMPLING'))
    save_csv(out/'task_auc_intervals.csv',taskrows)
    print('PASS: original 2,000-draw task-level AUC intervals.',flush=True)
    # Historical exact CAL selection for the already reported risk-budget grid.
    calrows=sorted([r for r in ep if r['split']=='CALIBRATION'],key=lambda r:r['episode_id'])
    cal=([r['episode_id'] for r in calrows],[float(r['guardian_score']) for r in calrows],[int(r['harm_indicator']) for r in calrows])
    curve=hs.curve(cal[1],cal[2]);riskrows=[];boundaries=[]
    assert config['risk_budget']['alphas']==['0.025','0.050','0.075','0.100']
    for alpha in config['risk_budget']['alphas']:
        for method in config['risk_budget']['methods']:
            selection=hs.select(curve,F(alpha),method,400,sum(cal[2]));threshold=F(selection['threshold_exact_fraction'])
            boundaries.append(dict(alpha=alpha,method=method,cutoff_exact=str(threshold),cutoff=float(threshold),cal_executed=selection['executed_count'],cal_harmful_executed=selection['harmful_executed_count'],cal_empirical_risk=selection['empirical_risk_exact'],cal_crc_bound=selection['crc_corrected_bound_exact']))
            for name,c in cohorts.items():
                e=np.array([int(F.from_float(float(s))<threshold) for s in c['score']])
                riskrows.append(dict(cohort=name,alpha=alpha,method=method,N=len(e),executed=int(e.sum()),harmful_executed=int((e*c['harm']).sum()),ARR=float(e.mean()),HERISK=float((e*c['harm']).mean()),cutoff_exact=str(threshold),scope='ARCHIVED_POST_PRIMARY_RISK_BUDGET_SENSITIVITY'))
    save_csv(out/'risk_budget_boundaries.csv',boundaries);save_csv(out/'risk_budget_results.csv',riskrows)
    print('PASS: archived risk-budget sensitivity; frozen primary policy unchanged.',flush=True)
    # Run the retained full historical function with only its output interfaces rebound.
    sizecfg=config['calibration_size']
    assert sizecfg['seed']==2026092504 and sizecfg['sample_sizes']==[100,200,300]
    assert sizecfg['replicates_each']==500 and sizecfg['full_cal_replicates']==1 and sizecfg['full_cal_reference']==400
    hs.ANALYSIS=out/'calibration_size';hs.save=save_csv;hs.json_save=save_json
    rep,size_summary=hs.calibration_size(cal,cohorts)
    assert len(rep)==3002 and len(size_summary)==48
    print('PASS: original CAL-size sensitivity (500 subsamples at 100/200/300; full-CAL reference).',flush=True)
    mixture=config['composition'];assert mixture['pool_n']==2400
    pool=[r for split in ('TEST','INDEPENDENT_CONFIRMATION') for r in sorted([r for r in ep if r['split']==split],key=lambda r:r['episode_id'])]
    compositions=[]
    for spec in mixture['specifications']:
        w=hs.weights([r['task_type'] for r in pool],[r['market_regime'] for r in pool],spec,mixture)
        for p in ('G0','G1','G2','G3'):
            e=np.array([permits[r['episode_id'],p] for r in pool]);h=np.array([int(r['harm_indicator']) for r in pool])
            compositions.append(dict(specification=spec['name'],policy=p,HERISK=float(np.dot(w,h*e)/w.sum()),ARR=float(np.dot(w,e)/w.sum()),scope='POST_PRIMARY_IN_SUPPORT_REWEIGHTING'))
    save_csv(out/'composition_reweighting.csv',compositions)
    print('PASS: archived task/regime composition reweighting; no new episodes.',flush=True)
    # Reconstruct inference using frozen TRAIN scaling and archived fitted coefficients.
    tc=config['tau'];feature_rows=read(root/'data/tau_features.csv')
    assert len(feature_rows)==4000 and [r['episode_id'] for r in feature_rows]==[r['episode_id'] for r in ep]
    fields=tc['numeric_features'];categories=tc['action_categories']
    X=np.array([[float(F(r[k])) for k in fields] for r in feature_rows],dtype=float)
    mean=np.array(tc['train_only_mean']);scale=np.array(tc['train_only_scale'])
    assert np.isfinite(X).all() and np.isfinite(mean).all() and np.all(scale>0)
    Z=np.hstack(((X-mean)/scale,np.eye(len(categories))[[categories.index(r['canonical_action']) for r in ep]]))
    assert tc['scaled_feature_order']==['numeric__'+k for k in fields]+['action__action_type_'+k for k in categories]
    calidx=np.array([i for i,r in enumerate(ep) if r['split']=='CALIBRATION'])
    loss=np.array([float(F(r['avoidable_loss'])) for r in ep]);taurows=[];taucal=[]
    for tau in tc['thresholds']:
        coef={r['feature']:float(r['coefficient']) for r in tc['coefficients'] if float(r['tau_H'])==float(tau)}
        score=expit(Z@np.array([coef[n] for n in tc['scaled_feature_order']])+coef['intercept'])
        labels=(loss>=float(tau)).astype(int);cv=hs.curve(score[calidx].tolist(),labels[calidx].tolist())
        selected={}
        for p,m in [('G2','G2_NAIVE'),('G3','G3_CRC')]:
            original=next(r for r in tc['archived_cutoffs'] if float(r['tau_H'])==float(tau) and r['method']==p)
            selected[p]=F(original['cutoff_exact'])
            actual=hs.select(cv,F(tc['alpha']),m,400,int(labels[calidx].sum()))
            # Exact equality only; no clipping, tolerance, threshold repair, or refitting.
            assert F(actual['threshold_exact_fraction'])==selected[p], 'TAU_ARCHIVED_CUTOFF_MISMATCH'
            assert actual['harmful_executed_count']==int(original['cal_harmful_executed'])
            taucal.append(dict(tau_H=tau,method=p,cutoff_exact=str(selected[p]),cal_harmful=int(labels[calidx].sum()),cal_harmful_executed=actual['harmful_executed_count'],scope='ARCHIVED_TAU_COEFFICIENT_REPLAY'))
        for split,name in [('TEST','TEST'),('INDEPENDENT_CONFIRMATION','CONFIRMATION')]:
            ix=np.array([i for i,r in enumerate(ep) if r['split']==split]);h=labels[ix]
            allowed={'G0':np.ones(len(ix),dtype=int),'G1':np.array([int(float(F(ep[i]['actual_exposure_fraction']))<=.10) for i in ix]),**{p:np.array([int(F.from_float(float(score[i]))<threshold) for i in ix]) for p,threshold in selected.items()}}
            for p,e in allowed.items():
                taurows.append(dict(tau_H=tau,split=name,regime=p,N=len(ix),harm_count=int(h.sum()),executed_count=int(e.sum()),harmful_executed_count=int((e*h).sum()),ARR=float(e.mean()),HERISK=float((e*h).mean()),scope='ARCHIVED_TAU_COEFFICIENT_REPLAY_NO_FIT'))
    save_csv(out/'tau_calibration.csv',taucal);save_csv(out/'tau_results.csv',taurows)
    print('PASS: archived tau sensitivity, six exact cutoffs; no training/refit performed.',flush=True)
    result=dict(status='PASS',primary_confidence_intervals='PASS',task_auc_intervals='PASS',calibration_size='PASS',risk_budget='PASS',composition_reweighting='PASS',tau_coefficient_replay='PASS',training_performed=False,network_calls=0,analysis_scope=config['analysis_scope'],not_verified=['Historical online LLM response regeneration','Original training/refit execution is not rerun; archived fitted parameters are used'])
    save_json(out/'sensitivity_summary.json',result)
    return result
