cd .. && ls


for m in "cifar10" "cifar100" "cinic10" "covid"
do
    for j in 1 2 3
    do
        for i in 0.1 0.3 0.5
        do
            python main.py --method fedopt  --thread_number 10 --dataset $m --client_number 100  --comm_round 500 --partition_alpha $i --client_sample 0.1
        done
    done
done
