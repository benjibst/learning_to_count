#base=/home/benni/dev/learning_to_count_data/vids
base=/home/benjamin/learning_to_count_data/vids

streamlink --stdout "$1" best | ffmpeg -i pipe:0 -c copy $base/$2
