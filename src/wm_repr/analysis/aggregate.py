"""Mean and standard error over training seeds of per-seed results."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def seed_summary(rows,groups,metrics,seed='seed'):
    """One observation per training seed; missing metrics remain explicitly counted."""
    if rows.duplicated(groups+[seed]).any():
        raise ValueError('Duplicate rows for a training seed; aggregate within each seed first')
    output=[]
    for key,block in rows.groupby(groups,dropna=False,sort=True):
        key=key if isinstance(key,tuple) else (key,)
        for metric in metrics:
            values=pd.to_numeric(block[metric],errors='coerce').to_numpy(float)
            values=values[np.isfinite(values)]
            n=len(values);sem=float(values.std(ddof=1)/np.sqrt(n)) if n>1 else np.nan
            output.append(dict(zip(groups,key),metric=metric,n=n,n_missing=len(block)-n,
                               mean=float(values.mean()) if n else np.nan,sem=sem))
    return pd.DataFrame(output)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--group-by',nargs='+',required=True)
    ap.add_argument('--metrics',nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=seed_summary(pd.read_csv(args.input),args.group_by,args.metrics)
    args.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.output,index=False)


if __name__=='__main__':main()
