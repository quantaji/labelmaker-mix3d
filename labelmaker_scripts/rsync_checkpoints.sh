source_dir=/cluster/project/cvg/labelmaker/labelmaker-mix3d/saved
target_dir=/home/guangda/repos/labelmaker-mix3d/saved

rsync -r --checksum -v -e ssh \
    guanji@euler.ethz.ch:$source_dir/* \
    $target_dir
