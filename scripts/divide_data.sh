# create a training folder
base="training/"
mkdir -p "$base"

# loop over subfolders of datasets
for dir in "../caltech256/256_ObjectCategories/"*
do
  # get the subfolder basename
  subdir=$(basename "$dir")

  # create a subfolder under training having the same basename
  mkdir -p "$base$subdir"

  # count the number of files in the current subfolder
  count=$(ls "$dir" | wc -l)

  # calculate the pourcentage to mv
  tenpercent=$(expr $count '*' 20 '/' 100)

  # move 20%
  ls "$dir" | gshuf -n "$tenpercent" | xargs -I {} mv "$dir"/{} "$base$subdir"

done
