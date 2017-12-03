:~/Workspace/ssd$ ./build/tools/caffe train -solver="models/input_hand/solver.prototxt" --gpu 0 --log_dir=jobs/input_hand --weights=models/input_hand/BodyFaceRegions_300x300_iter_120000.caffemodel
./build/tools/caffe train -solver="models/input_hand/solver.prototxt" --gpu 0 --log_dir=jobs/input_hand --snapshot=models/input_hand/BodyFaceHand_300x300_iter_21000.solverstate
