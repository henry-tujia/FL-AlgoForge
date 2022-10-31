cd .. && ls

for j in 1 2 3
do
    for i in 'moon'
    do
        python main.py --method $i  --thread_number 5 --dataset cifar10 --client_number 100 
    done
done