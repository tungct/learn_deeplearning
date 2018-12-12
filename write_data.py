import csv

def read_csv_file(file):
    with open(file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        datas = []
        for row in reader:
            datas.append(row)
    return datas

datas = read_csv_file("dantri_conv.csv")
count = 0
with open("data_10k.txt", "a") as myfile:
    for i in range(len(datas)):
        if count < 10:
            if datas[i] != [] or datas[i] != [' # ']:
                try:
                    myfile.write(datas[i][0])
                    count +=1
                except:
                    continue