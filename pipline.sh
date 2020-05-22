total=330000 # the total number of data points
k=100000 # the chunksize for astminer. Large chunksize may lead to memory out
ii=$(((total - total%k)/k+1))

# - extract comments from dataset
# - build a vocabulary for natural language
# - generate code files for astminer
a= rm -rf data/path_files
a= mkdir data/path_files
a= python scripts/nl_extract.py --k $k

# - use astminer to extract AST path from each code file
for ((i=0; i<=$ii; i++))
do
    a= java -jar cli-0.3.jar pathContexts --lang java --project data/path_files/train_$((k*i)) --output data/path_data/train_$i
done
a= java -jar cli-0.3.jar pathContexts --lang java --project data/path_files/test --output data/path_data/test

# - merge the output of each chunk
a= python scripts/share_vocab.py -t data/path_data/train_0/java -v data/path_data/train_1/java -o data/path_data/train/java --merge_vocab True
for ((i=2; i<$ii; i++))
do
    a= python scripts/share_vocab.py -t data/path_data/train/java -v data/path_data/train_$i/java -o data/path_data/train/java --merge_vocab True
done
a= python scripts/share_vocab.py -t data/path_data/train/java -v data/path_data/test/java -o data/path_data/test/java

#
a= python scripts/prepare_dataset.py -d data/path_data/train/java -o data/processed

# - model training
a= python main.py --pool_size 1000 --lr 0.0001 --batch_size 64 --emb_size 128 -e 200 --with_cuda 1

