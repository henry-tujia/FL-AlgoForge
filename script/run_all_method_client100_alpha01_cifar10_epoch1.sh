cd .. && ls

for j in 1 2 3
do
    for i in 'fedavg' 'fedrs' 'fedbalance' 'fedrod'  'moon'  'fedprox'
    do
        python main.py --method $i  --thread_number 10 --dataset cifar10 --client_number 100  --comm_round 100 --partition_alpha 0.1 --client_sample 0.1 --epochs 1
    done
done
