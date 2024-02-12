for d in 500 1000 2500 5000 7500 10000
do
    for lr in 0.01 0.001 0.0001 0.00001
    do
        for number_close_points in 10 50 100 150 200 500
        do
            for number_of_choices_training in 50 100 500 1000 2500 5000 7500 10000
            do
                for trial in 0 1 2
                do
                    (( time python3 models\Lidar_Seg\main.py --d "$d" --lr "$lr" --number_close_points "$number_close_points" --number_of_choices_training "$number_of_choices_training" --trial "$trial" ) 2>&1 ) | tee Lidar_Seg;
                done
            done
        done
    done
done