echo "Extracting TESSy data"
# assumes zip is in same folder as this script
mkdir -p raw
unzip AMR_TEST.zip && mv AMR_TEST.csv raw/AMR_TEST.csv
rm AMR_TEST.zip
rm AMR_TEST.zip:Zone.Identifier