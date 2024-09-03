echo "Downloading data from NCBI"

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

mkdir -p raw
# Note: Link may need to be updated, check NCBI for the latest link
wget -O raw/NCBI.tsv \
https://ftp.ncbi.nlm.nih.gov/pathogen/Results/Escherichia_coli_Shigella/PDG000000004.4469/Metadata/PDG000000004.4469.metadata.tsv