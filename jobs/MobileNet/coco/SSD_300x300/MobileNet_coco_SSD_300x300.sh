cd /home/asgdemo/Workspace/ssd
./build/tools/caffe train \
--solver="models/MobileNet/coco/SSD_300x300/solver.prototxt" \
--snapshot="models/MobileNet/coco/SSD_300x300/MobileNet_coco_SSD_300x300_iter_50000.solverstate" \
--gpu 2,3 2>&1 | tee jobs/MobileNet/coco/SSD_300x300/MobileNet_coco_SSD_300x300.log
