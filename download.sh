base=$LTC_DATA/vids

streamlink --stdout "$1" best | ffmpeg -i pipe:0 -c copy $base/$2
