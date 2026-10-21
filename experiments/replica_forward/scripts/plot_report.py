#!/usr/bin/env python3
"""Plot exported MOK metrics; CPU only, Matplotlib optional for benchmarking."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--metrics',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
rows=list(csv.DictReader(a.metrics.open()))
known=[('dsv4_pro','mxfp8','DSv4 Pro\nMXFP8'),('dsv4_pro','bf16','DSv4 Pro\nBF16'),('glm52','bf16','GLM 5.2\nBF16'),('qwen35_397b','bf16','Qwen3.5 397B\nBF16')]
lookup={(r['model'],r['precision'],r['profile']):r for r in rows}
cases=[c for c in known if any(m==c[0] and prec==c[1] for m,prec,_ in lookup)]
for m,prec in sorted({(m,prec) for m,prec,_ in lookup}):
    if (m,prec) not in {(c[0],c[1]) for c in cases}:cases.append((m,prec,m+'\n'+prec.upper()))
common=set.intersection(*[{profile for m,p,profile in lookup if m==model and p==prec} for model,prec,_ in cases])
if not common:raise SystemExit('No common profile across cases; select matching inputs.')
chosen='ECHO' if 'ECHO' in common else sorted(common)[0]
x=np.arange(len(cases));width=.24
fig,axes=plt.subplots(1,2,figsize=(13,5.4))
top=0
for offset,field,label,color in [(-width,'original_routed_tflops','OriginalReady','#4d7656'),(0,'serial_routed_tflops','Serial weight dispatch','#c3935e'),(width,'fused_routed_tflops','Fused weight dispatch','#466da8')]:
    values=[float(lookup[m,prec,chosen][field]) for m,prec,_ in cases]
    top=max(top,max(values))
    bars=axes[0].bar(x+offset,values,width,label=label,color=color)
    axes[0].bar_label(bars,fmt='%.0f',fontsize=8,padding=3)
axes[0].set_ylabel('Effective routed TFLOPS / GPU')
axes[0].set_ylim(0,top*1.22)
axes[0].set_title('Throughput: '+chosen+'-like synthetic placement')
hidden_values=[]
for offset,profile,color in [(-width,'ECHO','#c3935e'),(0,'UltraEP','#856da2'),(width,'MoonEP','#4d8d8c')]:
    values=[float(lookup[m,prec,profile]['recovered_serial_pct']) if (m,prec,profile) in lookup else np.nan for m,prec,_ in cases]
    hidden_values.extend(v for v in values if np.isfinite(v))
    bars=axes[1].bar(x+offset,values,width,label=profile+'-like',color=color)
    axes[1].bar_label(bars,labels=[f'{v:.0f}%' if np.isfinite(v) else '' for v in values],fontsize=8,padding=3)
axes[1].set_ylabel('Recovered serial overhead (%)')
axes[1].set_ylim(min(0,min(hidden_values)-8),max(100,max(hidden_values)+10))
axes[1].set_title('(Serial - Fused) / (Serial - OriginalReady)')
for ax in axes:
    ax.set_xticks(x,[label for _,_,label in cases])
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(fontsize=8,loc='upper right')
hardware=rows[0]['gpu'];ep=rows[0]['ep'];tokens=rows[0]['tokens']
fig.suptitle(f'MOK Forward | {hardware} | EP{ep}, {tokens} tokens/rank',fontsize=12)
fig.text(.5,.015,'Kernel time includes BF16 shared expert; routed FLOPs exclude shared. Plans are synthetic proxies, not real planner replay.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.055,1,.94))
a.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.output,dpi=180)
fig.savefig(a.output.with_suffix('.svg'))
svg=a.output.with_suffix('.svg')
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
print(a.output)
