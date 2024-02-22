for f in 0 1 2 3 4 5 6
do
    (( time python3 models\Lidar_Seg\main.py --features "$f" ) 2>&1 ) | tee Lidar_Seg;
done