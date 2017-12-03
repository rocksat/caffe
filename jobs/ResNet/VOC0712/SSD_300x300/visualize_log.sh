# visualize_log.sh
python tools/extra/parse_log.py jobs/ResNet/VOC0712/SSD_300x300/ResNet_VOC0712_SSD_300x300.log .
gnuplot -persist jobs/ResNet/VOC0712/SSD_300x300/gnuplot_commands