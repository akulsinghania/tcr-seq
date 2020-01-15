# %matplotlib inline
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from collections import defaultdict

matplotlib.rc('figure', figsize=(10., 8.))
matplotlib.rc('axes', labelsize='xx-large')
matplotlib.rc('xtick', labelsize='large')
matplotlib.rc('ytick', labelsize='large')


tcrseq = pd.read_csv('TCR_all_with_expansion.tsv.gz',
                     sep='\t', index_col=0, dtype={'donor_id':np.unicode_})


tcrseq.head()


##################################################################################################################
##################################################################################################################
#### Task 1: Get number of clones covering 80% of repertoire in different samples
#### First for every sample we compute how many clonotypes needed to cover 80%

cols = ['donor_id', 'stim', 'repl']
clones = {}
grpd = tcrseq.groupby(cols)
for gr in grpd.groups:
    tcrs = grpd.get_group(gr)
    freqs = tcrs['frequency'].sort_values(ascending=False).values
    # tcrs.to_csv('total_clones_' + gr[0] + '_' + gr[1] + '_' + str(gr[2]) + '.txt', sep='\t')
    N = (np.cumsum(freqs)>0.8).nonzero()[0][0]
    clones[gr] = N



with open('Number_of_clones_covering_80percent_repertoire.txt', 'w') as f:
    print(clones, file=f)


####  Now for one donor lets plot number of clonotypes covering 80% of productive repertoire N.B.
#### Now actual plotting









##################################################################################################################
##################################################################################################################
#### Task 2: Investigate how ex-vivo repertoire is affected by Stim

cols = ['donor_id', 'stim', 'repl']
grpd = tcrseq.groupby(cols)

plt.clf()
for ind, (donor) in enumerate([('3450'), ('3460'), ('3486'), ('3489'), ('3529'), ('3530')]):
    ax = plt.subplot(3, 2, ind+1)
    stim_freqs = grpd.get_group((donor, 'MTB300', 1))
    # significant will be those with Padj<0.05 and at least double alteration of frequency
    signif = stim_freqs[(stim_freqs.Padj<0.05)&(np.log2(stim_freqs.OR).abs()>1)]
    # all the other clonotypes
    notsignif = stim_freqs.loc[stim_freqs.index.difference(signif.index)]
    xmax = 0
    # Plot rearrangements with unchanged frequencies
    logOR = -np.log2(notsignif['OR'])
    # We also should add random jitter to the values
    stuck = logOR.abs()<0.5
    logOR.where(-stuck, logOR+0.01*(logOR.max()-logOR.min())*np.random.randn(logOR.shape[0]), inplace=True)
    clp = 50 # Clipping pvalues to 10e-50
    logPval = -(np.log10(notsignif['FisherP']).clip(lower=-clp))
    # Adding jitter
    logPval.where(-stuck, logPval+0.005*clp*np.abs(np.random.randn(logPval.shape[0])), inplace=True)
    # Now scatter plot
    ax.scatter(logOR, np.abs(logPval), c='none', s=20, edgecolors='blue')
    xmax = max(xmax, logOR.max(), logOR.abs().max())
    # Plot rearrangements with changed frequencies
    # no jitter needed
    logOR = -np.log2(signif['OR'])
    logPval = -(np.log10(signif['FisherP']).clip(lower=-50))
    logOR.to_csv(donor + '_MTB300_Repl1_vs_CD4_logOR.txt', sep='\t')
    logPval.to_csv(donor + '_MTB300_Repl1_vs_CD4_logPval.txt', sep='\t')
    ax.scatter(logOR, np.abs(logPval), c='none', s=20, edgecolors='red')
    xmax = max(xmax, logOR.max(), logOR.abs().max())
    # ax.set_xlabel('enriched pre vac <- log2 OR -> enriched post vac')
    ax.set_ylabel('-log2 p-value')
    xlim = np.max(ax.get_xlim())
    ax.set_xlim(-xmax*1.3, xmax*1.3)
    ax.text(0.95, 0.1, '{:s}'.format(donor),
           transform=ax.transAxes,
           ha='right', va='top')


plt.tight_layout()
plt.show()




##################################################################################################################
##################################################################################################################
#### Task 3: Investigate how ex-vivo repertoire is affected by Stim (with Syn-specific)

# Finding those with frequency affected by Syn
expanded_stim = tcrseq[(tcrseq.stim=="Syn")\
                          &(tcrseq.Padj<0.05)\
                          &(np.log2(tcrseq.OR).abs()>1)]\
                          .sort_values('OR')

syn_spec = defaultdict(list)
# Grouping tcrseq data by (donors) combination
tcrseq_by_donor = tcrseq.set_index('rearrangement')\
                              .loc[list(set(expanded_stim.rearrangement))]\
                              .groupby(['donor_id'])
for grp in tcrseq_by_donor.groups:
    print('processing group', grp)
    dnrdf = tcrseq_by_donor.get_group(grp)
    syn_rt = set(dnrdf[(dnrdf.stim=='Syn')&(dnrdf.FisherP<0.05)].index)
    for rt in syn_rt:
        rtdf = dnrdf.loc[[rt]]
        _Syn = rtdf[(rtdf.stim=='Syn')&(rtdf.OR<1)]
        _PT = rtdf[(rtdf.stim=='PT')&(rtdf.OR<1)]
        _MTB300 = rtdf[(rtdf.stim=='MTB300')&(rtdf.OR<1)]
        # we have 4 samples where a clonotype could expand against Syn:
        # 2 replicas pre vac and 2 replicas post vac
        # Those expanding in at least 2 of four will be considered Syn-specific
        if np.sum(_Syn['Padj']<0.05)>=2 and np.sum(_PT['Padj']<0.05)==0 and np.sum(_MTB300['Padj']<0.05)==0:
            syn_spec[grp].append(rt)



#### Now almost the same code as in Task 2 but making large dots for Syn-specific clonotypes
cols = ['donor_id', 'stim', 'repl']
grpd = tcrseq.groupby(cols)
plt.clf()

for ind, (donor) in enumerate([('3450'), ('3460'), ('3486'), ('3489'), ('3529'), ('3530')]):
    ax = plt.subplot(3, 2, ind+1)
    #ax = plt.gca()
    stim_freqs = grpd.get_group((donor, 'Syn', 1))
    signif = stim_freqs[(stim_freqs.Padj<0.05)&(np.log2(stim_freqs.OR).abs()>1)]
    notsignif = stim_freqs.loc[stim_freqs.index.difference(signif.index)]
    xmax = 0
    # Plot rearrangements with unchanged frequencies
    logOR = -np.log2(notsignif['OR'])
    stuck = logOR.abs()<0.5
    logOR.where(-stuck, logOR+0.01*(logOR.max()-logOR.min())*np.random.randn(logOR.shape[0]), inplace=True)
    clp = 50
    logPval = -(np.log10(notsignif['FisherP']).clip(lower=-clp))
    logPval.where(-stuck, logPval+0.005*clp*np.abs(np.random.randn(logPval.shape[0])), inplace=True)
    ax.scatter(logOR, np.abs(logPval), c='none', s=20, edgecolors='blue')
    xmax = max(xmax, logOR.max(), logOR.abs().max())
    # Plot altered rearrangements
    for tp in signif.itertuples():
        if tp.rearrangement in syn_spec[(donor)]:
            ax.scatter(-np.log(tp.OR), min(-np.log(tp.FisherP), 50), s=100, edgecolors='red', c='none')
        else:
            ax.scatter(-np.log(tp.OR), min(-np.log(tp.FisherP), 50), s=0, edgecolors='red', c='none')
    # Plot rearrangements with unchanged frequencies
    logOR = -np.log2(signif['OR'])
    logPval = -(np.log10(signif['FisherP']).clip(lower=-50))
    ax.scatter(logOR, np.abs(logPval), c='none', s=0, edgecolors='red')
    xmax = max(xmax, logOR.max(), logOR.abs().max())
    # ax.set_xlabel('enriched pre vac <- log2 OR -> enriched post vac')
    ax.set_ylabel('-log2 FT P value')
    xlim = np.max(ax.get_xlim())
    ax.set_xlim(-xmax*1.3, xmax*1.3)
    ax.text(0.95, 0.1, '{:s}'.format(donor),
           transform=ax.transAxes,
           ha='right', va='top')

plt.tight_layout()
plt.show()





##################################################################################################################
##################################################################################################################
#### Task 5: Overlap clones expanded in Syn and MTB stimulus
#### First collecting the data necessary for plotting in a DataFrame

cols = ['donor_id', 'stim', 'repl']
grpd = tcrseq.groupby(cols)

signif_query = 'Padj < 0.05 and OR < 1'
# idx has levels (donor, stim, prepost_vac)
# i.e. like in grpd but with replica dropped
idx = pd.MultiIndex.from_tuples(sorted(grpd.groups.keys()),
                                names=['D', 'S', 'R'])\
                   .droplevel(2).unique()
# Dropping everything except Syn and MTB
idx = idx.drop(set(idx.get_level_values(1)).difference({'Syn', 'MTB300'}), level=1)

expanded = {}
for dnr, stim in idx:
    _e1 = set(grpd.get_group((dnr, stim, 1)).query(signif_query)['rearrangement'])
    _e2 = set(grpd.get_group((dnr, stim, 2)).query(signif_query)['rearrangement'])
    PT_r1 = set(grpd.get_group((dnr, 'PT', 1)).query(signif_query)['rearrangement'])
    PT_r2 = set(grpd.get_group((dnr, 'PT', 2)).query(signif_query)['rearrangement'])
    PT_expanded = PT_r1.union(PT_r2)
    # expanded are those which are both replicas of stim but subtracting those
    # which expand in any PT replica
    expanded[(dnr, stim)] = _e1.intersection(_e2).difference(PT_expanded)

# diffs will be a DataFrame with necessary data
diffs = pd.DataFrame(index=idx.droplevel(1).unique(),
                     columns=['Syn_only', 'MTB300_only', 'Syn_shared', 'MTB300_shared'])

for donor in diffs.index:
    syn_reps = tcrseq[(tcrseq.donor_id==donor)&(tcrseq.stim=='Syn')]
    mtb_reps = tcrseq[(tcrseq.donor_id==donor)&(tcrseq.stim=='MTB300')]
    syn_only = expanded[(donor, 'Syn')].difference(expanded[(donor, 'MTB300')])
    shared = expanded[(donor, 'Syn')].intersection(expanded[(donor, 'MTB300')])
    mtb_only = expanded[(donor, 'MTB300')].difference(expanded[(donor, 'Syn')])
    diffs.loc[(donor), 'Syn_only'] = syn_reps.set_index('rearrangement').loc[syn_only].templates.sum()
    diffs.loc[(donor), 'Syn_shared'] = syn_reps.set_index('rearrangement').loc[shared].templates.sum()
    diffs.loc[(donor), 'MTB300_only'] = mtb_reps.set_index('rearrangement').loc[mtb_only].templates.sum()
    diffs.loc[(donor), 'MTB300_shared'] = mtb_reps.set_index('rearrangement').loc[shared].templates.sum()
    diffs.loc[(donor), 'Syn_tot'] = np.int64(syn_reps.templates.sum())
    diffs.loc[(donor), 'MTB300_tot'] = np.int64(mtb_reps.templates.sum())

# Here's how diffs looks like
diffs = diffs.astype(np.int64)
diffs







####
