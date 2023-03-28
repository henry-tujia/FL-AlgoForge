cd .. && ls

for j in 1 2 3
do

    for j in 0.1 0.3 0.5
    do
        python main.py --method fedbalance  --thread_number 4 --dataset covid --client_number 20  --comm_round 100 --partition_alpha $j --client_sample 0.2 --local_model lenet
    done

done
