source_dir=/media/hermann/data/labelmaker/ARKitScene_LabelMaker
target_dir=/cluster/project/cvg/labelmaker/ARKitScene_LabelMaker

# need:
# mesh.ply
# labels.txt
# point_lifted_mesh.ply

rsync -r --checksum -v -e ssh \
    --include='**/mesh.ply' \
    --include='**/labels.txt' \
    --include='**/point_lifted_mesh.ply' \
    --include='*/' \
    --exclude='*' \
    $source_dir/* \
    guanji@euler.ethz.ch:$target_dir
