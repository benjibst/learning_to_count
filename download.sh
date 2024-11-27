streamlink --stdout "$1" $3 | ffmpeg -i pipe:0 -c copy $2
