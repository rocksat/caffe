# visualize_log.sh
refresh_log() {
  while true; do
    python tools/extra/parse_log.py jobs/MobileNet/VOC0712/SSD_300x300/MobileNet_VOC0712_SSD_300x300.log jobs/MobileNet/VOC0712/SSD_300x300/
    sleep 5
  done
}
refresh_log &
sleep 1
gnuplot gnuplot_commands
