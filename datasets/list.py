import os 
path_root  = "mvtec_anomaly_detection" 
list  = os .listdir(path_root)

path_true = []
for i in list:
	path_cate = os.path.join(path_root,i)
	
	if os.path.isdir(path_cate) ==True:
		path_train = os.path.join(path_cate,'train','good')
		list_img = os.listdir(path_train)
		print(path_train)
		for i in list_img:
			path = os.path.join(path_train,i)
			path_true.append(path)

fo = open("train.flist", "w")
for i in range(len(path_true)):
    
    fo.write(path_true[i])
    fo.write('\n')
    # flist.append(path)

# print(flist)
fo.close()
