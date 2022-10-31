cd .. && ls

for j in 1 2 3
do
    for i in 'fedavg' 'fedrs' 'fedbalance' 'fedrod'  'moon'  'fedprox'
    do
        python main.py --method $i  --thread_number 2 --dataset cifar10 --client_number 20  --comm_round 100 --partition_alpha 0.3 --client_sample 0.2
    done
done
