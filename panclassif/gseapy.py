import pandas as pd
import gseapy as gs
import os
import warnings

def gsea(homepath):
    '''
    Parameters
    ----------
    homepath : str
    Path where you want to save all the generated files 
    and folders. 
        
    Return
    ------
    None

    Outputs
    -------
    Generate a directory named enrichr within home directory 
    and two plots of gene enrichment analysis using the 
    selected genes from panclassif.
    '''
    warnings.filterwarnings("ignore")
    
    # Create directory 'enrichr' if it doesn't exist
    directory = "enrichr"
    parent_dir = homepath
    path = os.path.join(parent_dir, directory)
    if not os.path.exists(path):
        os.mkdir(path)
        
    # Read gene list from CSV file
    gene = pd.read_csv(homepath + "/std_npy/unique_genes_with_frequency.csv", header=None)
    gl = list(gene[0])  # Extract gene list from DataFrame
    
    # Perform enrichment analysis using DisGeNET gene sets
    enr = gs.enrichr(gene_list=gl, description='Disease', gene_sets='DisGeNET', outdir=homepath + '/enrichr')
    
    # Import plotting functions
    from gseapy.plot import barplot, dotplot
    
    # Generate and save barplot and dotplot
    barplot(enr.res2d, title='DisGeNET', cutoff=0.2, ofname=homepath + '/enrichr/DisGeNET_barplot.png')
    dotplot(enr.res2d, title='DisGeNET', cmap='viridis_r', cutoff=0.2, ofname=homepath + '/enrichr/DisGeNET_dotplot.png')