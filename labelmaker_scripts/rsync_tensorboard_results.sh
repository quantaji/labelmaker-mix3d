source_dir=/cluster/project/cvg/labelmaker/labelmaker-mix3d/saved
target_dir=$HOME/exp_results

mkdir -p $target_dir

rsync -r --checksum -v -e ssh \
    --include='**/events.out.tfevents.*' \
    --include='*/' \
    --exclude='*' \
    guanji@euler.ethz.ch:$source_dir/* \
    $target_dir
