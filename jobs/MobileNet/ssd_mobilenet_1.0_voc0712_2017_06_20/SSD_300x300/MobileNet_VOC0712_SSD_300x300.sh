cd /home/asgdemo/Workspace/ssd
./build/tools/caffe train \
--solver="models/MobileNet/VOC0712/SSD_300x300/solver.prototxt" \
--weights="models/MobileNet/mobilenet.caffemodel" \
--gpu 2,3 2>&1 | tee jobs/MobileNet/VOC0712/SSD_300x300/MobileNet_VOC0712_SSD_300x300.log
