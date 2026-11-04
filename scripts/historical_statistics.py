"""Retained historical statistical functions used by the offline reproduction entry."""
import csv,hashlib,json,math
from fractions import Fraction as F
from decimal import Decimal,localcontext
from statistics import NormalDist
import numpy as np

def require(ok,code):
    if not ok:raise RuntimeError(code)

def wilson(k,n):
    require(type(k) is int and type(n) is int and n>0 and 0<=k<=n,'INVALID_BINOMIAL_COUNTS')
    z=NormalDist().inv_cdf(.975)
    if k==0:return [0.,z*z/(n+z*z)]
    if k==n:return [n/(n+z*z),1.]
    p=k/n
    divisor=1+z*z/n
    center=(p+z*z/(2*n))/divisor
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/divisor
    return [max(0.,center-radius),min(1.,center+radius)]


def policy_metrics(harm,execute,ieu,payment,is_payment):
    h=np.asarray(harm,dtype=np.int64); e=np.asarray(execute,dtype=np.int64)
    u=np.asarray(ieu,dtype=np.float64); pay=np.asarray(payment,dtype=np.float64); mask=np.asarray(is_payment,dtype=bool)
    n=len(h)
    require(n>0 and all(len(x)==n for x in (e,u,pay,mask)),'METRIC_SHAPE')
    require(np.isin(h,[0,1]).all() and np.isin(e,[0,1]).all() and np.isfinite(u).all() and np.isfinite(pay).all(),'METRIC_INPUT')
    positive=int(h.sum()); negative=n-positive; count=int(e.sum()); k=int((h*e).sum()); safe=count-k
    npay=int(mask.sum())
    result={'N':n,'Harm':positive,'Nonharm':negative,'executed':count,'abstained':n-count,
        'harmful_executed':k,'harmful_abstained':positive-k,'safe_executed':safe,'safe_abstained':negative-safe,
        'HERISK':k/n,'ARR':count/n,'GovernanceBurden':(n-count)/n,
        'SARR':safe/negative if negative else None,'HABR':(positive-k)/positive if positive else None,
        'MeanIEU':math.fsum((u*e).tolist())/n,'MedianIEU':float(np.median(u*e)),
        'PaymentN':npay,'PaymentCompletion':math.fsum((pay[mask]*e[mask]).tolist())/npay if npay else None}
    result['Wilson95']={'HERISK':wilson(k,n),'ARR':wilson(count,n),'SARR':wilson(safe,negative) if negative else None,'HABR':wilson(positive-k,positive) if positive else None}
    return result


def bootstrap(harm,executions,ieu,payment,is_payment,protocol):
    h=np.asarray(harm,dtype=np.int64); e=np.asarray(executions,dtype=np.int64)
    u=np.asarray(ieu,dtype=np.float64); p=np.asarray(payment,dtype=np.float64); mask=np.asarray(is_payment,dtype=bool)
    n=len(h); b=protocol['replicates']; seed=protocol['seed']
    require(e.shape==(n,4) and protocol['sample_size']==n and protocol['paired_across_all_policies'] and protocol['replacement'],'BOOTSTRAP_PROTOCOL')
    require(protocol['rng']=='numpy.random.PCG64' and protocol['quantile_method']=='linear','BOOTSTRAP_IMPLEMENTATION')
    # Exactly one shared index vector per replicate for every policy and metric.
    indices=np.random.Generator(np.random.PCG64(seed)).integers(0,n,size=(b,n),dtype=np.int64)
    pair=e[:,3,None]-e[:,:3]
    arrays={'HERISK':pair*h[:,None],'ARR':pair,'MeanIEU':pair*u[:,None],'PaymentCompletion':pair*p[:,None]*mask[:,None]}
    denominator=mask[indices].sum(axis=1)
    require(np.all(denominator>0),'STOP_EMPTY_PAYMENT_BOOTSTRAP_REPLICATE')
    rows=[]
    for metric,values in arrays.items():
        for j,comparator in enumerate(('G0','G1_010','G2_005')):
            samples=values[indices,j].sum(axis=1)/(denominator if metric=='PaymentCompletion' else n)
            point=math.fsum(values[:,j].tolist())/(int(mask.sum()) if metric=='PaymentCompletion' else n)
            lo,hi=np.percentile(samples,protocol['percentiles'],method=protocol['quantile_method'])
            rows.append({'contrast':'G3_005 - '+comparator,'metric':metric,'point_estimate':point,'ci_lower':float(lo),'ci_upper':float(hi)})
    return {'rows':rows,'replicates':b,'seed':seed,'indices_sha256':hashlib.sha256(indices.astype('<i8',copy=False).tobytes()).hexdigest(),
        'zero_payment_replicates':0,'paired':True,'quantile_method':'linear'}


def exact_score(score):
    score = float(score)
    require(math.isfinite(score) and 0 <= score <= 1, "STOP_SCORE_CORRUPTION")
    return F.from_float(score)


def candidates(scores):
    # De-duplicate exact stored binary64 score values; evaluate their mathematical
    # complements without another binary64 rounding. This preserves r==1-lambda
    # ties even when floating 1-(1-r) would not equal r. No tolerance or clipping.
    return tuple(sorted({F(0), F(1)} | {1 - exact_score(s) for s in scores}))


def execute(score, lam):
    lam = F(lam)
    require(0 <= lam <= 1, "STOP_LAMBDA_RANGE")
    return exact_score(score) < 1 - lam


def risks(k, n):
    require(type(k) is int and type(n) is int and n > 0 and 0 <= k <= n, "INVALID_COUNTS")
    return F(k, n), F(k + 1, n + 1)


def cap(alpha, n, corrected):
    value = F(str(alpha)) * (n + 1 if corrected else n)
    return min(n, value.numerator // value.denominator - int(corrected))


def decimal_text(value):
    value = F(value)
    # Candidate lambdas/thresholds are dyadic rationals and have finite decimals.
    with localcontext() as context:
        context.prec = 1200
        return format(Decimal(value.numerator) / Decimal(value.denominator), "f")


def curve(scores, labels):
    require(len(scores) == len(labels) > 0 and all(type(y) is int and y in (0, 1) for y in labels), "STOP_CALIBRATION_LABEL_INPUT")
    rows = []
    for lam in candidates(scores):
        decisions = [execute(s, lam) for s in scores]
        count = sum(decisions)
        k = sum(y for y, decision in zip(labels, decisions) if decision)
        empirical, corrected = risks(k, len(scores))
        rows.append({"lambda": lam, "threshold": 1 - lam, "executed_count": count,
                     "ARR": F(count, len(scores)), "harmful_executed": k, "empirical_risk": empirical, "crc_bound": corrected})
    for earlier, later in zip(rows, rows[1:]):
        require(all(earlier[n] >= later[n] for n in ("executed_count", "ARR", "harmful_executed", "empirical_risk", "crc_bound")), "STOP_MONOTONICITY")
    return rows


def select(rows, alpha, method, n, harm):
    require(method in ("G2_NAIVE", "G3_CRC"), "UNKNOWN_METHOD")
    metric = "empirical_risk" if method == "G2_NAIVE" else "crc_bound"
    eligible = [i for i, row in enumerate(rows) if row[metric] <= F(alpha)]
    index = eligible[0] if eligible else len(rows) - 1
    row = rows[index]
    require(row["lambda"] == 1 if not eligible else row[metric] <= F(alpha), "SELECTOR_FAILURE")
    if index > 0 and eligible:
        require(rows[index - 1][metric] > F(alpha), "STOP_NOT_MINIMAL_LAMBDA")
    k, count = row["harmful_executed"], row["executed_count"]
    safe = n - harm
    return {"method": method, "alpha": alpha, "lambda_hat": decimal_text(row["lambda"]),
        "execution_threshold": decimal_text(row["threshold"]), "lambda_exact_fraction": str(row["lambda"]),
        "threshold_exact_fraction": str(row["threshold"]), "executed_count": count, "abstained_count": n-count,
        "ARR": float(row["ARR"]), "harmful_executed_count": k, "harmful_abstained_count": harm-k,
        "safe_executed_count": count-k, "safe_abstained_count": safe-(count-k),
        "empirical_risk": float(row["empirical_risk"]), "empirical_risk_exact": str(row["empirical_risk"]),
        "crc_corrected_bound": float(row["crc_bound"]), "crc_corrected_bound_exact": str(row["crc_bound"]),
        "SARR": (count-k)/safe if safe else None, "HABR": (harm-k)/harm if harm else None,
        "candidate_index": index, "previous_candidate": None if index == 0 else {
            "lambda": decimal_text(rows[index-1]["lambda"]), "selection_risk_exact": str(rows[index-1][metric]), "infeasible": rows[index-1][metric] > F(alpha)},
        "empty_set_fallback": not bool(eligible)}


def fast_choose(scores, labels, alpha, method):
    # Descending score groups correspond exactly to ascending candidate lambdas.
    n = len(scores); maximum = cap(alpha, n, method == "G3_CRC")
    ordered = sorted(zip(scores, labels), reverse=True)
    k = sum(labels)
    if k <= maximum and all(s < 1 for s in scores):
        return F(0)
    i = 0
    while i < n:
        score = ordered[i][0]
        while i < n and ordered[i][0] == score:
            k -= ordered[i][1]; i += 1
        if k <= maximum:
            return 1 - F.from_float(score)
    return F(1)


def metrics(c, execute):
    return policy_metrics(c["harm"], execute, c["ieu"], c["payment"], c["mask"])


def calibration_size(cal,cohorts):
    ids,scores,labels=cal
    rng=np.random.Generator(np.random.PCG64(2026092504))
    specs=[(n,rep,rng.choice(400,size=n,replace=False).tolist()) for n in (100,200,300) for rep in range(500)]
    specs.append((400,0,list(range(400))))
    # Persist all indices before any held-out outcome is inspected.
    json_save(ANALYSIS/"calibration_subsample_indices.json",[{"n_cal":n,"replicate_id":rep,"indices":ix} for n,rep,ix in specs])
    out=[]
    for n,rep,ix in specs:
        ss=[scores[i] for i in ix]; yy=[labels[i] for i in ix]
        l2=fast_choose(ss,yy,"0.05","G2_NAIVE");l3=fast_choose(ss,yy,"0.05","G3_CRC")
        for cohort,c in cohorts.items():
            e2=np.array([int(F.from_float(float(x))<1-l2) for x in c["score"]],dtype=np.int8)
            e3=np.array([int(F.from_float(float(x))<1-l3) for x in c["score"]],dtype=np.int8)
            if np.any(e3>e2):raise RuntimeError("STOP_POLICY_NESTEDNESS_BUG")
            m2=metrics(c,e2);m3=metrics(c,e3);dis=e2!=e3
            out.append(dict(n_cal=n,replicate_id=rep,cohort=cohort,G2_lambda=decimal_text(l2),G3_lambda=decimal_text(l3),
                G2_threshold=decimal_text(1-l2),G3_threshold=decimal_text(1-l3),lambda_gap=float(l3-l2),threshold_gap=float(l3-l2),
                G2_G3_disagreement_N=int(dis.sum()),G2_G3_disagreement_rate=float(dis.mean()),harmful_disagreement_N=int((dis*c["harm"]).sum()),
                G2_HERISK=m2["HERISK"],G3_HERISK=m3["HERISK"],G2_ARR=m2["ARR"],G3_ARR=m3["ARR"],
                Delta_HERISK=m3["HERISK"]-m2["HERISK"],Delta_ARR=m3["ARR"]-m2["ARR"],
                correction=f"1/{n+1}",G2_kmax=cap("0.05",n,False),G3_kmax=cap("0.05",n,True)))
    save(ANALYSIS/"calibration_size_replicates.csv",out)
    summary=[]
    for n in (100,200,300,400):
        for cohort in cohorts:
            rr=[r for r in out if r["n_cal"]==n and r["cohort"]==cohort]
            for key in ("lambda_gap","threshold_gap","G2_G3_disagreement_rate","harmful_disagreement_N","Delta_HERISK","Delta_ARR"):
                vals=[r[key] for r in rr]
                summary.append(dict(n_cal=n,cohort=cohort,metric=key,replicates=len(vals),median=float(np.median(vals)),
                    q25=float(np.percentile(vals,25)),q75=float(np.percentile(vals,75)),p05=float(np.percentile(vals,5)),p95=float(np.percentile(vals,95)),
                    interpretation="FROZEN_FULL_CAL_REFERENCE" if n==400 else "SUBSAMPLE_SENSITIVITY_DISTRIBUTION"))
    save(ANALYSIS/"calibration_size_sensitivity_summary.csv",summary)
    return out,summary


def weights(tasks,regimes,spec,cfg):
    require(len(tasks)==len(regimes),'WEIGHT_SHAPE')
    tp=dict(zip(cfg['task_order'],map(F,spec['task'])));rp=dict(zip(cfg['regime_order'],map(F,spec['regime'])))
    require(sum(tp.values())==sum(rp.values())==1 and all(x>0 for x in list(tp.values())+list(rp.values())),'TARGET_PROBABILITY')
    require(set(tasks)<=set(tp) and set(regimes)<=set(rp),'UNKNOWN_CATEGORY')
    return np.asarray([float(tp[t]*rp[r]/F(cfg['reference_joint_probability'])) for t,r in zip(tasks,regimes)])

