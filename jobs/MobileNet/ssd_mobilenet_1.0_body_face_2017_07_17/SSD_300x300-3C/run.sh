./build/tools/caffe train --solver="models/MobileNet/ssd_mobilenet_1.0_body_face_2017_07_17/SSD_300x300-3C/solver.prototxt"  --gpu 2,3 --log_dir=jobs/MobileNet/ssd_mobilenet_1.0_body_face_2017_07_17/SSD_300x300-3C --weights="models/BodyFaceRegions-VOC0712_170404_161114-3C/BodyFaceRegions_300x300_iter_120000.caffemodel"
