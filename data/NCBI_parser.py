import os
import numpy as np
import pandas as pd

from pathlib import Path

# user-defined functions
from utils import filter_gene_counts

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

def preprocess_NCBI(path,
                    save_path = None,
                    include_phenotype: bool = False, 
                    ab_names_to_abbr: dict = None,
                    exclude_antibiotics: list = None,
                    threshold_year: int = None,
                    exclude_genotypes: list = None,
                    exclude_assembly_variants: list = None,
                    exclusion_chars: list = None,
                    gene_count_threshold: int = None,
                    ):

    NCBI_data = pd.read_csv(path, sep='\t', low_memory=False) 
    cols = ['collection_date', 'geo_loc_name', 'AMR_genotypes_core']
    if include_phenotype:
        cols += ['AST_phenotypes']
    df = NCBI_data[cols]
    df = df.rename(columns={'AMR_genotypes_core': 'genotypes'})
    df = df[df['genotypes'].notnull()] # filter out missing genotypes
    if include_phenotype:
        print("Parsing phenotypes...")
        df = df.rename(columns={'AST_phenotypes': 'phenotypes'})
        indices = df[df['phenotypes'].notnull()].index 
        df.loc[indices, 'phenotypes'] = df.loc[indices, 'phenotypes'].str.split(',')
        df.loc[indices, 'phenotypes'] = df.loc[indices, 'phenotypes'].apply(lambda x: [p for p in x if p.split("=")[1] in ['R', 'S']])
        name_to_abbr = {k.casefold(): v for k, v in ab_names_to_abbr.items()}
        df.loc[indices, 'phenotypes'] = df.loc[indices, 'phenotypes'].apply(
            lambda x: [name_to_abbr[p.split("=")[0].casefold()] + "_" + p.split("=")[1] for p in x if 
                       p.split("=")[0].casefold() in name_to_abbr.keys() and p.split("=")[1] in ['R', 'S']]
        )
        if exclude_antibiotics:
            print(f"Filtering out antibiotics: {exclude_antibiotics}")
            df.loc[indices, 'phenotypes'] = df.loc[indices, 'phenotypes'].apply(
                lambda x: [p for p in x if p.split("_")[0] not in exclude_antibiotics]
            )
        df['num_ab'] = df['phenotypes'].apply(lambda x: len(x) if isinstance(x, list) else np.nan)
        df = df[df['num_ab'] != 0]
        df['num_ab'] = df['num_ab'].replace(np.nan, 0)
        print(f"Number of isolates with phenotype info: {df[df['num_ab'] > 0].shape[0]:,}")
    
    #### geo_loc_name -> country 
    alternative_nan = ['not determined', 'not collected', 'not provided', 'Not Provided', 'OUTPATIENT',
                       'missing: control sample', 'Not collected', 'Not Collected', 'not available', '-']
    df.loc[:,'geo_loc_name'] = df['geo_loc_name'].replace(alternative_nan, np.nan)
    
    # Remove regional information
    df.loc[:,'geo_loc_name'] = df['geo_loc_name'].str.split(',').str[0]
    df.loc[:,'geo_loc_name'] = df['geo_loc_name'].str.split(':').str[0] 
    df = df.rename(columns={'geo_loc_name': 'country'})
    df.loc[:,'country'] = df['country'].replace(
        {'United Kingdom': 'UK', 'United Arab Emirates': 'UAE', 'Democratic Republic of the Congo': 'DRC',
         'Republic of the Congo': 'DRC', 'Czechia': 'Czech Republic', 'France and Algeria': 'France'})
    
    ##### collection_date -> year
    alternative_nan = ['missing']
    df.loc[:,'collection_date'] = df['collection_date'].replace(alternative_nan, np.nan)
    df.loc[:,'collection_date'] = df['collection_date'].str.split('-').str[0]
    df.loc[:,'collection_date'] = df['collection_date'].str.split('/').str[0]
    df = df.rename(columns={'collection_date': 'year'})
    
    #### Parse genotypes
    df['genotypes'] = df['genotypes'].str.split(',')
    print("Parsing genotypes...")
    print(f"Number of isolates before parsing: {df.shape[0]:,}")
    
    if threshold_year:
        indices = df[df['year'].astype(float) < threshold_year].index
        print(f"Removing {len(indices):,} isolates with year < {threshold_year}")
        df.drop(indices, inplace=True)
        
    df['genotypes'] = df['genotypes'].apply(lambda x: list(set([g.strip() for g in x]))) # remove whitespace and duplicates
    if exclude_genotypes: 
        print(f"Removing genotypes: {exclude_genotypes}")
        df['genotypes'] = df['genotypes'].apply(lambda x: [g for g in x if g not in exclude_genotypes]) 
    
    if exclude_assembly_variants: # Examples: ["=PARTIAL", "=MISTRANSLATION", "=HMM"]
        print(f"Removing genotypes with assembly variants: {exclude_assembly_variants}")
        df['genotypes'] = df['genotypes'].apply(lambda x: [g for g in x if not g.endswith(tuple(exclude_assembly_variants))]) 
    df = df[df['genotypes'].apply(lambda x: len(x) > 0)] # Remove any rows where genotypes are empty
    
    if exclusion_chars:
        # old_genotypes = df['genotypes'].copy() # save old genotypes for later
        for char in exclusion_chars:
            print(f"Splitting genotypes by '{char}', removing it and everything after it")
            df['genotypes'] = df['genotypes'].apply(lambda x: list(set([g.split(char)[0] for g in x])))
            
    # Remove cases where there is both a genotype and an assembly variant
    assembly_chars = ['=PARTIAL', '=MISTRANSLATION', '=HMM', '=PARTIAL_END_OF_CONTIG']
    df['genotypes'] = df['genotypes'].apply(
        lambda x: list(set(x) - set([g for g in x if g.endswith(tuple(assembly_chars)) and g.split("=")[0] in x])))
          
    df['num_genotypes'] = df['genotypes'].apply(lambda x: len(set(x)))
    df['num_point_mutations'] = df['genotypes'].apply(lambda x: len([g for g in x if '=POINT' in g]))
    
    if gene_count_threshold:
        df = filter_gene_counts(df, gene_count_threshold)

    # Exclude cases where there is only one genotype and no other info
    df = df[~((df['num_genotypes'] == 1) & (df['country'].isnull()) & (df['year'].isnull()))]
    print(f"Number of isolates after parsing: {df.shape[0]:,}")
    if include_phenotype:
        print(f"Number of isolates with phenotype info after parsing: {df[df['num_ab'] > 0].shape[0]:,}")
    if save_path:
        print(f"Saving to {save_path}")
        df.to_pickle(save_path)
    
    return df