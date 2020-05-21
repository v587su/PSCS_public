#change k to prevent from memory out
#total=330000
#k=100000
total=15
k=5
ii=$(((total - total%k)/k))
a= rm -rf data/path_files
a= mkdir data/path_files
a= python scripts/nl_extract.py --k $k

for ((i=0; i<=$ii; i++))
do
    a= java -jar cli-0.3.jar pathContexts --lang java --project data/path_files/train_$((k*i)) --output data/path_data/train_$i
done
a= java -jar cli-0.3.jar pathContexts --lang java --project data/path_files/test --output data/path_data/test
echo "path files generated！"
a= python scripts/share_vocab.py -t data/path_data/train_0/java -v data/path_data/train_1/java -o data/path_data/train/java --merge_vocab True
for ((i=2; i<$ii; i++))
do
    a= python scripts/share_vocab.py -t data/path_data/train/java -v data/path_data/train_$i/java -o data/path_data/train/java --merge_vocab True
done
a= python scripts/share_vocab.py -t data/path_data/train/java -v data/path_data/test/java -o data/path_data/test/java

a= python scripts/prepare_dataset.py -d data/path_data/train/java -o data/processed
a= python main.py --pool_size 1000 --lr 0.0001 --batch_size 64 --emb_size 128 -e 200 --with_cuda 0

