cd .. && ls

for j in 1 2 3
do
    for i in 0.1 0.3 0.5
    do
        python main.py --method fedbalance  --thread_number 5 --dataset cifar10 --client_number 100  --comm_round 500 --partition_alpha $i --client_sample 0.1
    done
done
