#!/usr/bin/env python3
"""Run-scoped degradation plots and paired daily-relative gaps."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from equity_metrics import load_grid, write_metrics

ORDER=["daily","weekly","mask","sparse20","sparse30","sparse40"]
COLORS={"ghana":"#2c5f8a","italy_primary":"#2c8a5f","italy_secondary":"#8a2c5f"}

def draw(df, out, label):
    methods=sorted(df.method.unique())
    gaps=write_metrics(df,out/f"{label}_paired_gaps.csv")
    for metric in ["degradation","equity"]:
        fig,axes=plt.subplots(1,len(methods),figsize=(6*len(methods),4.8),squeeze=False)
        for ax,method in zip(axes[0],methods):
            subset=df[df.method==method]
            regimes=[r for r in ORDER if r in set(subset.regime) and (metric=="degradation" or r!="daily")]
            regimes+=sorted(set(subset.regime)-set(ORDER))
            if not regimes:
                ax.text(.5,.5,"No sparse regimes",ha="center",transform=ax.transAxes)
            for country in sorted(subset.country.unique()):
                if metric=="degradation":
                    rows=subset[subset.country==country]
                    values=[rows.loc[rows.regime==r,"mase"].median() for r in regimes]
                    ax.plot(range(len(regimes)),values,marker="o",label=country,color=COLORS[country])
                else:
                    rows=gaps[(gaps.method==method)&(gaps.country==country)]
                    stats=rows.groupby("regime").mase_gap_pct.agg(["median","count"])
                    countries=sorted(subset.country.unique()); i=countries.index(country); width=.8/len(countries)
                    values=[stats.loc[r,"median"] if r in stats.index else np.nan for r in regimes]
                    ax.bar(np.arange(len(regimes))-.4+width*(i+.5),values,width,label=country,color=COLORS[country])
            ax.set_xticks(range(len(regimes)),regimes,rotation=30)
            ax.set_title(method)
            ax.set_ylabel("Median MASE" if metric=="degradation" else "Median paired MASE gap vs daily (%)")
            ax.axhline(0,color="grey",lw=.7)
            ax.spines[["top","right"]].set_visible(False)
            ax.legend(fontsize=8,frameon=False)
        title="Ghana vs Italy PRIMARY" if label=="primary" else "Italy SECONDARY: robustness only, not cross-country equity"
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out/f"fig_{metric}_{label}.png",dpi=160)
        plt.close(fig)
    df.groupby(["country","method","regime"]).mase.agg(["median","count"]).to_csv(out/f"{label}_degradation_data.csv")

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ghana",required=True,nargs="+",type=Path)
    ap.add_argument("--italy-primary",required=True,nargs="+",type=Path)
    ap.add_argument("--italy-secondary",nargs="+",type=Path,default=[])
    ap.add_argument("--outdir",required=True,type=Path)
    ap.add_argument("--allow-legacy-italy-label",action="store_true",help="Explicitly assign legacy italy rows to the arm specified by each input flag")
    a=ap.parse_args()
    paths={"ghana":a.ghana,"italy_primary":a.italy_primary,"italy_secondary":a.italy_secondary}
    frames={}
    provenance={}
    for country,files in paths.items():
        if not files: continue
        df=pd.concat([load_grid(p,country,a.allow_legacy_italy_label) for p in files],ignore_index=True)
        if df.duplicated(["country","method","origin","seed","regime"]).any():
            raise ValueError(f"{country}: overlapping fit keys across input files")
        frames[country]=df
        provenance[country]=[{"path":str(p.resolve()),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
        print(df.groupby(["country","method","regime"]).size().to_string())
    if a.outdir.exists() and any(a.outdir.iterdir()):
        raise ValueError("Use an empty/new --outdir for this run; stale plots must not survive a failed rerun")
    a.outdir.mkdir(parents=True,exist_ok=True)
    draw(pd.concat([frames["ghana"],frames["italy_primary"]],ignore_index=True),a.outdir,"primary")
    if "italy_secondary" in frames: draw(frames["italy_secondary"],a.outdir,"secondary")
    (a.outdir/"inputs.json").write_text(json.dumps(provenance,indent=2))

if __name__=="__main__": main()
