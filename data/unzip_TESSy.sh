echo "Extracting TESSy data"

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"/raw

unzip AMR_TEST.zip
rm AMR_TEST.zip
rm AMR_TEST.zip:Zone.Identifier