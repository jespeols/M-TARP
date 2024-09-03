echo "Downloading data from NCBI"
mkdir -p raw
# Note: Link may need to be updated, check NCBI for the latest link
wget -O raw/NCBI.tsv \
https://ftp.ncbi.nlm.nih.gov/pathogen/Results/Escherichia_coli_Shigella/PDG000000004.4460/Metadata/PDG000000004.4460.metadata.tsv