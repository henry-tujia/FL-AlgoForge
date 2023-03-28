cd .. && ls

for j in 1 2 3
do
    for i in 0.1 0.3 0.5
    do
        for m in 'fedavg' 'fedrs' 'fedbalance' 'fedrod'  'moon'  'fedprox'
        do
            # echo $i $m
            python main.py --method $m  --thread_number 5 --dataset cifar100 --client_number 100  --comm_round 300 --partition_alpha $i --client_sample 0.1
        done
    done
done
