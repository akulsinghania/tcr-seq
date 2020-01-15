import os
from collections import defaultdict
import re
import zipfile
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import fdrcorrection
import pandas as pd


# cd Desktop/Parkinson_TCR/X_sectional/without_3507/without_MTB/data
# zip -r ../data.zip *


tcr_data = {}
columns = ['donor_id', 'stim', 'repl']
sampdata = pd.DataFrame(index=[], columns=columns)


##################################################################################################################
##################################################################################################################
#### Part 1. Reading Adaptive TCR-seq data from two zip archives and gathering metadata
#### Read the first batch of TCR-seq data

with zipfile.ZipFile('data.zip') as z:
    for fl in z.namelist():
        # e.g. 1497_1_MTB_1.tsv removing extension
        samp = os.path.splitext(fl)[0]
        donor, arbit, stim_repl = re.match(r'(\d+)-(\d)-(.+)_TCRB', samp).groups()
        if stim_repl == 'XV-CD4':
            stim = 'CD4'
            repl = 1
        else:
            stim = stim_repl[:-2]
            repl = int(stim_repl[-1])
        sampdata.loc[samp, columns] = donor, stim, repl
        with z.open(fl) as fh:
            tcr_data[samp] = pd.read_csv(fh, sep='\t', low_memory=False)


sampdata.sort_values(columns, inplace=True)

sampdata


##################################################################################################################
##################################################################################################################
#### Merge all samples into single DataFrame

# Merging all samples into one big dataframe and grabbing
# only necessary columns from big output files of
grpd = sampdata.groupby(columns)
dfs = []
for gr, idx in grpd.groups.items():
    # combination of values in columns uniquely identifies a sample
    assert len(idx) == 1
    samp = grpd.get_group(gr).index[0]
    columns2take = ['rearrangement', 'amino_acid', 'templates',
                    'total_templates', 'frequency',
                    'productive_entropy', 'frame_type',
                    'v_family', 'v_gene', 'v_allele',
                    'd_family', 'd_gene', 'd_allele',
                    'j_family', 'j_gene', 'j_allele']
    tcrs = tcr_data[samp][columns2take]
    # Dropping out-of-frame rearrangements
    tcrs = tcrs[tcrs.frame_type=="In"]
    # Getting the frequencies of rearrangements corrected for removed out-of-frame
    tcrs['frequency'] = tcrs['frequency'] / tcrs['frequency'].sum()
    # Instantiating columns
    for col in columns:
        tcrs[col] = np.nan
    # now we set necessary columns of tcrs DataFrame with the tuple
    # identifying the sample
    tcrs[columns] = gr
    tcrs['sample'] = idx[0]
    dfs.append(tcrs)

tcrseq = pd.concat(dfs)
tcrseq['indeks'] = np.arange(len(tcrseq))
tcrseq.set_index('indeks', inplace=True)

tcrseq.head()



##################################################################################################################
##################################################################################################################
####  Part 2. Find clones which are expanded in 14-day cultures
#### Get rearrangement frequencies as a dictionary


# N.B. columns is still ['donor_id', 'stim', 'repl']
# i.e. a combination uniquely identifying one sample
grpd = tcrseq.groupby(columns)

# Need a dictionary for fast access of a rearrangement frequency
# in a given donor, pre or post vaccination i.e.
# dictionary d-> (donor, pre/post) -> rearrangement -> # of templates
ex_vivo_freqs = {}
# Also on the fly gather total # of in-frame templates in every sample
total_templates = {}

for gr in grpd.groups:
    if gr[1] != 'CD4':
        total_templates[gr] = grpd.get_group(gr)['templates'].sum()
    else:
        s = grpd.get_group(gr).set_index('rearrangement')['templates']
        d = s.to_dict()
        dd = defaultdict(lambda: 1, d)
        ex_vivo_freqs[(gr[0])] = dd
        total_templates[gr] = s.sum()



##################################################################################################################
##################################################################################################################
####  Calculate p-values of expansion in 14day culture using Fisher-exact test

# dictionary pvals -> group -> index -> (odds-ratio, p-value)
# where group is a tuple (donor_id, pre/post, stim, repl, batch)
pvals = defaultdict(dict)

for gr in grpd.groups:
    if gr[1]=='CD4':
        continue
    print('Processing group', gr)
    for r in grpd.get_group(gr).itertuples():
        # Getting values of the contingency table in the form
        # [[x11, x12],
        #  [x21, x22]]
        # where x11 templates of the clonotype ex vivo
        #       x12 total templates in donor ex vivo excluding the clonotype
        #       x21 templates of the clonotype in 14d culture
        #       x22 total templates in 14d culture excluding the clonotype
        x11 = ex_vivo_freqs[(gr[0])][r.rearrangement]
        x12 = total_templates[(gr[0], 'CD4', 1)] - x11
        x21 = r.templates
        x22 = total_templates[gr] - r.templates
        p = fisher_exact([[x11, x12],[x21, x22]], alternative='two-sided')
        pvals[gr][r.Index] = p




##################################################################################################################
##################################################################################################################
#### Now inserting these p-values into the main DataFrame
for gr in pvals:
    ii = list(pvals[gr].keys());
    tcrseq.loc[ii, 'FisherP'] = [pvals[gr][i][1] for i in ii]
    tcrseq.loc[ii, 'OR'] = [pvals[gr][i][0] for i in ii]


# Correcting 14-day cultures
for gr in pvals:
    if gr[1] == 'CD4':
        continue
    df = grpd.get_group(gr)
    padj = fdrcorrection(df.FisherP)[1]
    tcrseq.loc[df.index, 'Padj'] = padj


tcrseq.to_csv('TCR_all_with_expansion.tsv.gz', sep='\t')
