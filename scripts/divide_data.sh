# This script is to divide the dataset into 80% 20%
echo "If you run this script on linux, change gshuf with shuf "
base_folder="../caltech256/256_ObjectCategories_test/"

mkdir -p "$base_folder"

for dir in "../caltech256/256_ObjectCategories/"*
do

  subdir=$(basename "$dir")

  mkdir -p "$base_folder$subdir"

  count=$(ls "$dir" | wc -l)

  tenpercent=$(expr $count '*' 20 '/' 100)

  ls "$dir" | shuf -n "$tenpercent" | xargs -I {} mv "$dir"/{} "$base_folder$subdir"

done
