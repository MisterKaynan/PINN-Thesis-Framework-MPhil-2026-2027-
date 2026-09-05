import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np
m = pd.read_csv('data/processed/ghana_national_master.csv', index_col=0, parse_dates=True)
fig, ax = plt.subplots(2,1, figsize=(9,5.2), sharex=True,
                       gridspec_kw={'height_ratios':[3,1], 'hspace':0.12})
ax[0].fill_between(m.index, m.daily_incidence, color='#c8d8e8', lw=0)
ax[0].plot(m.index, m.daily_incidence, color='#2c5f8a', lw=0.8)
for d in m.index[m.observed_simple==0]:
    ax[0].axvline(d, color='#d94f4f', alpha=0.16, lw=1.4)
ax[0].set_ylabel('Daily reported cases')
ax[0].set_title('Ghana COVID-19 national incidence with unreported days shaded (HERA, 12 Mar 2020 – 19 May 2021)',
                fontsize=10, loc='left')
h=[plt.Line2D([],[],color='#2c5f8a',lw=1.2,label='Reported incidence'),
   plt.Line2D([],[],color='#d94f4f',lw=4,alpha=0.3,label='No report (117 d, 27.0%)')]
ax[0].legend(handles=h, fontsize=8, frameon=False, loc='upper left')
gl = m.gap_length.where(m.gap_length>0, 0)
ax[1].bar(m.index, gl, color='#d94f4f', width=1.0)
ax[1].set_ylabel('Gap\nlength (d)', fontsize=8)
ax[1].set_xlabel('Date')
ax[1].annotate('14-day gap\n(15–28 Mar 2020)', xy=(pd.Timestamp('2020-03-21'),14),
               xytext=(pd.Timestamp('2020-05-05'),12.2), fontsize=7.5,
               arrowprops=dict(arrowstyle='->', lw=0.7, color='#555'))
for a in ax: a.spines[['top','right']].set_visible(False)
plt.savefig('results/figures/fig_data_quality.png', dpi=170, bbox_inches='tight')
print('ok')
