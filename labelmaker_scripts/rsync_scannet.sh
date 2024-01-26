# rsync scennet necessary files to a temporary folder
source_dir=/cluster/project/cvg/Shared_datasets/scannet
target_dir=/home/guangda/scannet_rsync

# need:
# rgb _vh_clean_2.ply
# instance *.aggregation.json
# instance [0-9].segs.json
# semantic label .labels.ply

rsync -r --checksum -v -e ssh \
    --include='scannetv2-labels.combined.tsv' \
    --include='**/*_vh_clean_2.ply' \
    --include='**/*.aggregation.json' \
    --include='**/*[0-9].segs.json' \
    --include='**/scene[0-9][0-9][0-9][0-9]_[0-9][0-9].txt' \
    --include='**/*.labels.ply' \
    --include='*/' \
    --exclude='*' \
    --exclude='**/data/' \
    --exclude='**/data_compressed/' \
    --exclude='**/instance-filt/' \
    --exclude='**/label-proc/' \
    guanji@euler.ethz.ch:$source_dir/* \
    $target_dir

mkdir /home/guangda/repos/labelmaker-mix3d/data/raw/scannet
ln -s $target_dir /home/guangda/repos/labelmaker-mix3d/data/raw/scannet/scannet

git clone https://github.com/ScanNet/ScanNet.git /home/guangda/repos/labelmaker-mix3d/data/raw/scannet/ScanNet
