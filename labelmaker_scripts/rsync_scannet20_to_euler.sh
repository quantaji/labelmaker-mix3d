source_dir=/home/guangda/repos/labelmaker-mix3d/data/processed/scannet
target_dir=/cluster/project/cvg/labelmaker/labelmaker-mix3d/data/processed/scannet
rsync -r --checksum -v -e ssh \
    $source_dir/* \
    guanji@euler.ethz.ch:$target_dir