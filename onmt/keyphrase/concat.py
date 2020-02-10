fw = open("/home/yingyi/Documents/output/kp20k/roberta-base/tokenized/kp20k_train_all.json", 'w')
from tqdm import tqdm
for i in tqdm(range(1,8)):
    f =open("/home/yingyi/Documents/output/kp20k/roberta-base/tokenized/train_"+str(i)+".json","r").readlines()
    for line in f:
        line = line.strip()
        fw.write(line+"\n")

fw.close()
