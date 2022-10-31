cd .. && ls

for j in 1 2 3
do
    for i in 'fedavg' 'fedrs' 'fedrod' 'fedbalance' 'moon'  'fedprox'
    do
        python main.py --method $i  --thread_number 4 --dataset cifar10 --client_number 20  --comm_round 200 --partition_alpha 0.1 --client_sample 0.2
    done
done
