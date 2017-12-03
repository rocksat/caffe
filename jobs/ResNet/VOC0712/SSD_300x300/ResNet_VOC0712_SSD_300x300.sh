cd /home/asgdemo/Workspace/ssd
./build/tools/caffe train \
--solver="models/ResNet/VOC0712/SSD_300x300/solver.prototxt" \
--snapshot="models/ResNet/VOC0712/SSD_300x300/ResNet_VOC0712_SSD_300x300_iter_8719.solverstate" \
--gpu 1 2>&1 | tee jobs/ResNet/VOC0712/SSD_300x300/ResNet_VOC0712_SSD_300x300.log
