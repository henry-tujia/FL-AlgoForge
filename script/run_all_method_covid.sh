cd .. && ls

for j in 1 2 3
do
    for i in 'fednova' 'fedopt'
    do
        for j in 0.1 0.3 0.5
        do
            python main.py --method $i  --thread_number 4 --dataset covid --client_number 20  --comm_round 100 --partition_alpha $j --client_sample 0.2
        done
    done
done
