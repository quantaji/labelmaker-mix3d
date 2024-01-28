source_dir=/home/guangda/scannet_rsync
target_dir=/home/quanta/Projects/labelmaker-mix3d/data/raw/scannet/scannet

rsync -r --checksum -v -e ssh \
    guangda@129.132.245.59:$source_dir/* \
    $target_dir
