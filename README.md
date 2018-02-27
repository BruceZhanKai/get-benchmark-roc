# get-cnn-feature

## Requirement

### caffe
### opencv

## Structure

```
—————./
|————get-benchmark-roc
|————Parameter.txt
|————lfw-resize-6000-1103/
||————pairs.txt
||————Aaron_Eckhart/
|||————Aaron_Eckhart_0001.jpg
|————Result/
||————lfw-resize-6000-1103-[][][][][][][]-2018-2-27/
|||————ACC.txt
|||————feature1.txt
|||————feature2.txt
|||————ratio.txt
|||————record.txt
|||————ROC.txt
|||————thres.txt
|————CNNModel/
```

## LFW benchmark SOP

### Step1 training
```
將 normalize 成 192x192 眼距 40 的 training images
放置根目錄自定義資料夾中
例如，"./dataset/001/pic1.jpg"
```
### Step2 training
```
執行 get-cnn-feature，得到 feature vector.txt & label.txt
```
### Step3 training
```
丟 TXT 給 Gary 去 train bayesian model
```
### Step4 testing
```
執行get-benchmark-roc，拿 train 好的 cnn model 及 bayes model 來 test 結果並存成 roc.TXT
```
### Step5 testing
```
將 roc.TXT 丟給 LFW 提供的 matlab code 去算 ROC 曲線
```












