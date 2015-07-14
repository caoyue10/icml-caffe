##Net
Transfer Task from ''Amazon'' Data to ''Dslr'' Data in Office Dataset

##Dataset
This model use images in ''Amazon'' as source domain and iamges in ''Dslr'' as target domain.

##Finetune
Please train this model based on ''CaffeNet'' model given by caffe

##Training
Training and testing mages are provided to ''caffe'' in ''train.txt'' and ''text.txt'' files individually.

In this ''caffe'' for transfer task, one should put source and target image paths of training data in the same ''train.txt'' file, in which source paths first and target paths follows. In addition, in ''train_val.prototxt'', one should give the parameter to tell the number of training source data.

The batch size in training procedure is fixed to 64, the first 32 of which are from source domain and the others from target domain.
