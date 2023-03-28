cd .. && ls

for j in 1 2 3
do
    for i in 0.1 0.3 0.5
    do
        python main.py --method fedbalance  --thread_number 5 --dataset cifar100 --client_number 100  --comm_round 300 --partition_alpha $i --client_sample 0.1
    done
done
