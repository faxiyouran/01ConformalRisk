"""Minimal offline interface to unchanged historical diagnostic functions."""
import csv,io
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SPLITS=("TEST","INDEPENDENT_CONFIRMATION")
TASKS=("CAPITAL_PRESERVATION","DAPP_APPROVAL","OUTBOUND_PAYMENT","PORTFOLIO_REBALANCE")
ACTIONS=("APPROVE","SWAP")
BUDGETS=(10,20,30)
SEED=20261102
REPLICATES=2000
def csv_bytes(rows,fields=None):
    rows=list(rows);s=io.StringIO(newline="")
    w=csv.DictWriter(s,fieldnames=fields or list(rows[0]),lineterminator="\n")
    w.writeheader();w.writerows(rows)
    return s.getvalue().encode("utf-8")


def write(path,content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(content,str):content=content.encode("utf-8")
    path.write_bytes(content)


def save_csv(path,rows):write(path,csv_bytes(rows))


def rng_for(experiment,split_index,action_index=0):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED,experiment,split_index,action_index])))


def intervals(values):return np.quantile(values,[.025,.975],method="linear").tolist()

