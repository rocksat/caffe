cd /home/asgdemo/Workspace/ssd
./build/tools/caffe train \
--solver="models/MobileNet/ssd_mobilenet_1.0_body_face_2017_07_17/SSD_300x300/solver.prototxt" \
--snapshot="models/MobileNet/ssd_mobilenet_1.0_body_face_2017_07_17/SSD_300x300/MobileNet_Body_Face_SSD_300x300_iter_20000.solverstate" \
# --weights="models/MobileNet/mobilenet.caffemodel" \
--gpu 2,3 2>&1 | tee jobs/MobileNet/ssd_mobilenet_1.0_body_face_2017_07_17/SSD_300x300/MobileNet_Body_Face_SSD_300x300.log \

