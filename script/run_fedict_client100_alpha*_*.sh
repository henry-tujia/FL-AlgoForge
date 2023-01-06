cd .. && ls

for j in 1 2 3
do
    for method in "loss" "logits" "probs"
    do
        for m in "cifar10" "cifar100" "cinic10" "covid"
        do
            for i in 0.1 0.3 0.5
            do
                python main.py --method fedict  --thread_number 10 --dataset $m --client_number 100  --comm_round 500 --partition_alpha $i --client_sample 0.1 --weight_method $method
            done
        done
    done
done
