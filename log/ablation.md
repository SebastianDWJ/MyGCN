去掉后面的attention:

=> Average test accuracy in 10-fold CV: 0.87715
=> Average test AUC in 10-fold CV: 0.89275
=> Average test sensitivity 0.9031, specificity 0.8676, F1-score 0.8818

=> Average test accuracy in 10-fold CV: 0.87141
=> Average test AUC in 10-fold CV: 0.88537
=> Average test sensitivity 0.8939, specificity 0.8760, F1-score 0.8810

=> Average test accuracy in 10-fold CV: 0.86797
=> Average test AUC in 10-fold CV: 0.88245
=> Average test sensitivity 0.8909, specificity 0.8613, F1-score 0.8744

=> Average test accuracy in 10-fold CV: 0.83812
=> Average test AUC in 10-fold CV: 0.87560
=> Average test sensitivity 0.8436, specificity 0.8975, F1-score 0.8634



去掉前面的attention：

=> Average test accuracy in 10-fold CV: 0.87830
=> Average test AUC in 10-fold CV: 0.89121
=> Average test sensitivity 0.8965, specificity 0.8803, F1-score 0.8867

=> Average test accuracy in 10-fold CV: 0.87371
=> Average test AUC in 10-fold CV: 0.89578
=> Average test sensitivity 0.8987, specificity 0.8676, F1-score 0.8817


不同层数GCN：
8:
=> Average test accuracy in 10-fold CV: 0.87830
=> Average test AUC in 10-fold CV: 0.88829
=> Average test sensitivity 0.9033, specificity 0.8762, F1-score 0.8863

7：
=> Average test accuracy in 10-fold CV: 0.88060
=> Average test AUC in 10-fold CV: 0.89818
=> Average test sensitivity 0.8918, specificity 0.8932, F1-score 0.8901

=> Average test accuracy in 10-fold CV: 0.88175
=> Average test AUC in 10-fold CV: 0.89580
=> Average test sensitivity 0.8986, specificity 0.8848, F1-score 0.8894

5：
=> Average test accuracy in 10-fold CV: 0.87371
=> Average test AUC in 10-fold CV: 0.89347
=> Average test sensitivity 0.9059, specificity 0.8592, F1-score 0.8803

=> Average test accuracy in 10-fold CV: 0.88060
=> Average test AUC in 10-fold CV: 0.89598
=> Average test sensitivity 0.9030, specificity 0.8762, F1-score 0.8874


4：
=> Average test accuracy in 10-fold CV: 0.86338
=> Average test AUC in 10-fold CV: 0.87577
=> Average test sensitivity 0.9027, specificity 0.8357, F1-score 0.8643

=> Average test accuracy in 10-fold CV: 0.87486
=> Average test AUC in 10-fold CV: 0.89690
=> Average test sensitivity 0.8969, specificity 0.8826, F1-score 0.8852



random:
=> Average test accuracy in 10-fold CV: 0.70608
=> Average test AUC in 10-fold CV: 0.71032
=> Average test sensitivity 0.7111, specificity 0.7822, F1-score 0.7388


站点
=> Average test accuracy in 10-fold CV: 0.71757
=> Average test AUC in 10-fold CV: 0.72777
=> Average test sensitivity 0.7341, specificity 0.7675, F1-score 0.7422

=> Average test accuracy in 10-fold CV: 0.71986
=> Average test AUC in 10-fold CV: 0.73165
=> Average test sensitivity 0.7294, specificity 0.7757, F1-score 0.7432

站点、年龄
=> Average test accuracy in 10-fold CV: 0.80253
=> Average test AUC in 10-fold CV: 0.83862
=> Average test sensitivity 0.8036, specificity 0.8440, F1-score 0.8212

=> Average test accuracy in 10-fold CV: 0.79334
=> Average test AUC in 10-fold CV: 0.82512
=> Average test sensitivity 0.8130, specificity 0.8052, F1-score 0.8064

站点、性别、年龄
=> Average test accuracy in 10-fold CV: 0.80712
=> Average test AUC in 10-fold CV: 0.85276
=> Average test sensitivity 0.8161, specificity 0.8294, F1-score 0.8206

=> Average test accuracy in 10-fold CV: 0.81171
=> Average test AUC in 10-fold CV: 0.86793
=> Average test sensitivity 0.8191, specificity 0.8570, F1-score 0.8278

=> Average test accuracy in 10-fold CV: 0.81515         2
=> Average test AUC in 10-fold CV: 0.84217
=> Average test sensitivity 0.8306, specificity 0.8250, F1-score 0.8254

=> Average test accuracy in 10-fold CV: 0.81286         1
=> Average test AUC in 10-fold CV: 0.85628
=> Average test sensitivity 0.8202, specificity 0.8527, F1-score 0.8296 

=> Average test accuracy in 10-fold CV: 0.82434         0.5
=> Average test AUC in 10-fold CV: 0.85278
=> Average test sensitivity 0.8577, specificity 0.8144, F1-score 0.8306




输入是 sites 、gender：
=> Average test accuracy in 10-fold CV: 0.88060
=> Average test AUC in 10-fold CV: 0.89282
=> Average test sensitivity 0.9183, specificity 0.8528, F1-score 0.8830

=> Average test accuracy in 10-fold CV: 0.87371
=> Average test AUC in 10-fold CV: 0.88807
=> Average test sensitivity 0.9086, specificity 0.8548, F1-score 0.8794

=> Average test accuracy in 10-fold CV: 0.86797
=> Average test AUC in 10-fold CV: 0.88825
=> Average test sensitivity 0.9016, specificity 0.8528, F1-score 0.8724

=> Average test accuracy in 10-fold CV: 0.87486
=> Average test AUC in 10-fold CV: 0.89001
=> Average test sensitivity 0.9037, specificity 0.8697, F1-score 0.8825

输入是sites：
=> Average test accuracy in 10-fold CV: 0.72331
=> Average test AUC in 10-fold CV: 0.73055
=> Average test sensitivity 0.7315, specificity 0.7737, F1-score 0.7506

=> Average test accuracy in 10-fold CV: 0.70953
=> Average test AUC in 10-fold CV: 0.71543
=> Average test sensitivity 0.7367, specificity 0.7287, F1-score 0.7264

=> Average test accuracy in 10-fold CV: 0.71297
=> Average test AUC in 10-fold CV: 0.72920
=> Average test sensitivity 0.7108, specificity 0.7990, F1-score 0.7479

输入是site、age：
=> Average test accuracy in 10-fold CV: 0.71986
=> Average test AUC in 10-fold CV: 0.72724
=> Average test sensitivity 0.7382, specificity 0.7605, F1-score 0.7436

=> Average test accuracy in 10-fold CV: 0.71757
=> Average test AUC in 10-fold CV: 0.72209
=> Average test sensitivity 0.7259, specificity 0.7865, F1-score 0.7476

=> Average test accuracy in 10-fold CV: 0.72216
=> Average test AUC in 10-fold CV: 0.72903
=> Average test sensitivity 0.7267, specificity 0.7926, F1-score 0.7526

输入是gender、age：
=> Average test accuracy in 10-fold CV: 0.70953
=> Average test AUC in 10-fold CV: 0.68309
=> Average test sensitivity 0.6892, specificity 0.8674, F1-score 0.7641

=> Average test accuracy in 10-fold CV: 0.70723
=> Average test AUC in 10-fold CV: 0.68689
=> Average test sensitivity 0.6998, specificity 0.8396, F1-score 0.7572

=> Average test accuracy in 10-fold CV: 0.69805
=> Average test AUC in 10-fold CV: 0.67972
=> Average test sensitivity 0.7061, specificity 0.8101, F1-score 0.7396


去掉a.att
=> Average test accuracy in 10-fold CV: 0.86682
=> Average test AUC in 10-fold CV: 0.88937
=> Average test sensitivity 0.9031, specificity 0.8463, F1-score 0.8729
去掉b.att
=> Average test accuracy in 10-fold CV: 0.86338
=> Average test AUC in 10-fold CV: 0.88052
=> Average test sensitivity 0.8832, specificity 0.8468, F1-score 0.8720
=> Average test accuracy in 10-fold CV: 0.86338
=> Average test AUC in 10-fold CV: 0.87577
=> Average test sensitivity 0.9027, specificity 0.8357, F1-score 0.8643
去掉a.att & b.att
=> Average test accuracy in 10-fold CV: 0.84918
=> Average test AUC in 10-fold CV: 0.86948
=> Average test sensitivity 0.8762, specificity 0.8568, F1-score 0.8620

=> Average test accuracy in 10-fold CV: 0.85534
=> Average test AUC in 10-fold CV: 0.87718
=> Average test sensitivity 0.8857, specificity 0.8482, F1-score 0.8644

卷积算子消融
普通GCN:
=> Average test accuracy in 10-fold CV: 0.81860
=> Average test AUC in 10-fold CV: 0.85090
=> Average test sensitivity 0.8468, specificity 0.8401, F1-score 0.8299

=> Average test accuracy in 10-fold CV: 0.83467
=> Average test AUC in 10-fold CV: 0.85134
=> Average test sensitivity 0.8639, specificity 0.8508, F1-score 0.8453

=> Average test accuracy in 10-fold CV: 0.84271
=> Average test AUC in 10-fold CV: 0.85962
=> Average test sensitivity 0.8694, specificity 0.8529, F1-score 0.8491

GraphConv：
=> Average test accuracy in 10-fold CV: 0.79334
=> Average test AUC in 10-fold CV: 0.82537
=> Average test sensitivity 0.8437, specificity 0.7932, F1-score 0.7908

=> Average test accuracy in 10-fold CV: 0.79334
=> Average test AUC in 10-fold CV: 0.80842
=> Average test sensitivity 0.8434, specificity 0.7823, F1-score 0.8016

=> Average test accuracy in 10-fold CV: 0.80023
=> Average test AUC in 10-fold CV: 0.83018
=> Average test sensitivity 0.8016, specificity 0.8442, F1-score 0.8175

GAT：
=> Average test accuracy in 10-fold CV: 0.79449
=> Average test AUC in 10-fold CV: 0.78200
=> Average test sensitivity 0.8257, specificity 0.8297, F1-score 0.8039

=> Average test accuracy in 10-fold CV: 0.79449
=> Average test AUC in 10-fold CV: 0.78628
=> Average test sensitivity 0.7871, specificity 0.8914, F1-score 0.8206

=> Average test accuracy in 10-fold CV: 0.79793
=> Average test AUC in 10-fold CV: 0.80681
=> Average test sensitivity 0.8016, specificity 0.8551, F1-score 0.8129

=> Average test accuracy in 10-fold CV: 0.79679
=> Average test AUC in 10-fold CV: 0.78517
=> Average test sensitivity 0.7948, specificity 0.8402, F1-score 0.8060

=> Average test accuracy in 10-fold CV: 0.80138     heads=3,dropout=0.2,concat=False,negative_slope=0.1, bias=bias
=> Average test AUC in 10-fold CV: 0.79985
=> Average test sensitivity 0.8427, specificity 0.7734, F1-score 0.7957

=> Average test accuracy in 10-fold CV: 0.80827     heads = 8
=> Average test AUC in 10-fold CV: 0.79521
=> Average test sensitivity 0.8549, specificity 0.7953, F1-score 0.8051

TAG：
=> Average test accuracy in 10-fold CV: 0.82664     
=> Average test AUC in 10-fold CV: 0.86127
=> Average test sensitivity 0.8258, specificity 0.8634, F1-score 0.8421

=> Average test accuracy in 10-fold CV: 0.82319     5
=> Average test AUC in 10-fold CV: 0.85445
=> Average test sensitivity 0.8347, specificity 0.8510, F1-score 0.8371

=> Average test accuracy in 10-fold CV: 0.81515     4
=> Average test AUC in 10-fold CV: 0.85888
=> Average test sensitivity 0.8448, specificity 0.8208, F1-score 0.8251

=> Average test accuracy in 10-fold CV: 0.83123     7
=> Average test AUC in 10-fold CV: 0.85246
=> Average test sensitivity 0.8507, specificity 0.8399, F1-score 0.8422

=> Average test accuracy in 10-fold CV: 0.83467     9
=> Average test AUC in 10-fold CV: 0.84014
=> Average test sensitivity 0.8812, specificity 0.8058, F1-score 0.8325

=> Average test accuracy in 10-fold CV: 0.86567     10
=> Average test AUC in 10-fold CV: 0.89354
=> Average test sensitivity 0.8775, specificity 0.8825, F1-score 0.8776

=> Average test accuracy in 10-fold CV: 0.85649      15
=> Average test AUC in 10-fold CV: 0.88063
=> Average test sensitivity 0.8630, specificity 0.8718, F1-score 0.8666

=> Average test accuracy in 10-fold CV: 0.84386     12
=> Average test AUC in 10-fold CV: 0.87195
=> Average test sensitivity 0.8678, specificity 0.8524, F1-score 0.8556

ARMA：
=> Average test accuracy in 10-fold CV: 0.79104
=> Average test AUC in 10-fold CV: 0.83268
=> Average test sensitivity 0.7994, specificity 0.8507, F1-score 0.8180

=> Average test accuracy in 10-fold CV: 0.80827
=> Average test AUC in 10-fold CV: 0.83569
=> Average test sensitivity 0.8079, specificity 0.8593, F1-score 0.8288

=> Average test accuracy in 10-fold CV: 0.80023
=> Average test AUC in 10-fold CV: 0.83166
=> Average test sensitivity 0.8218, specificity 0.8314, F1-score 0.8200

=> Average test accuracy in 10-fold CV: 0.80367
=> Average test AUC in 10-fold CV: 0.82219
=> Average test sensitivity 0.8238, specificity 0.8250, F1-score 0.8202

=> Average test accuracy in 10-fold CV: 0.80941
=> Average test AUC in 10-fold CV: 0.83217
=> Average test sensitivity 0.8454, specificity 0.7991, F1-score 0.8172

=> Average test accuracy in 10-fold CV: 0.80482
=> Average test AUC in 10-fold CV: 0.82417
=> Average test sensitivity 0.8290, specificity 0.8210, F1-score 0.8184

SSG:
=> Average test accuracy in 10-fold CV: 0.80138
=> Average test AUC in 10-fold CV: 0.82520
=> Average test sensitivity 0.8220, specificity 0.8421, F1-score 0.8197

=> Average test accuracy in 10-fold CV: 0.81171
=> Average test AUC in 10-fold CV: 0.82853
=> Average test sensitivity 0.8080, specificity 0.8932, F1-score 0.8402

=> Average test accuracy in 10-fold CV: 0.80941
=> Average test AUC in 10-fold CV: 0.80667
=> Average test sensitivity 0.8329, specificity 0.8443, F1-score 0.8225

=> Average test accuracy in 10-fold CV: 0.81171
=> Average test AUC in 10-fold CV: 0.82614
=> Average test sensitivity 0.8467, specificity 0.8079, F1-score 0.8185


ClusterGCNConv:   
=> Average test accuracy in 10-fold CV: 0.83123         2
=> Average test AUC in 10-fold CV: 0.86523
=> Average test sensitivity 0.8628, specificity 0.8225, F1-score 0.8393

=> Average test accuracy in 10-fold CV: 0.83238             1.5
=> Average test AUC in 10-fold CV: 0.84712
=> Average test sensitivity 0.8478, specificity 0.8484, F1-score 0.8445

=> Average test accuracy in 10-fold CV: 0.83812             3
=> Average test AUC in 10-fold CV: 0.86084
=> Average test sensitivity 0.8414, specificity 0.8760, F1-score 0.8555

=> Average test accuracy in 10-fold CV: 0.83467         4
=> Average test AUC in 10-fold CV: 0.84035
=> Average test sensitivity 0.8340, specificity 0.8868, F1-score 0.8534

=> Average test accuracy in 10-fold CV: 0.85649         3.5
=> Average test AUC in 10-fold CV: 0.89437
=> Average test sensitivity 0.8836, specificity 0.8459, F1-score 0.8613

=> Average test accuracy in 10-fold CV: 0.84845         3.3
=> Average test AUC in 10-fold CV: 0.86774
=> Average test sensitivity 0.8662, specificity 0.8609, F1-score 0.8593

=> Average test accuracy in 10-fold CV: 0.85419     3.5
=> Average test AUC in 10-fold CV: 0.87704
=> Average test sensitivity 0.8766, specificity 0.8608, F1-score 0.8646

EG:
