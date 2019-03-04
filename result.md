# Assignment 1
 
### Class:電腦視學特效
### Group:20
### Member:
楊政新(107065808)
黃中佑(103081045)
陳昱名(107062574)
 
 ### 1.Introduction 
本次作業一是運用深度學習GAN中較為特殊的CycleGAN模型，製作影像合成、置換，材質以及顏色轉換的效果。本組透過將範例的python source code 放入linux環境透過GPU上加速進行Training，共training完成夏冬轉換、蘋果橘子轉換、梵谷畫作與真實照片轉換三個model。
關於內文對模型的介紹將主要依據CHIEN-YI WANG所發表解釋GAN與CycleGAN兩個Model功能的文章做整理[(CHIEN-YI WANG, 2017)](https://data-sci.info/2017/05/25/cyclegan/)。

 **1.1 GAN**
 深度學習的生成式對抗網路模型(Generative Adversarial Network, GAN)，如圖1-1所示，有Discriminator network與Generator network兩個模型，Generator network的工作是做出人造的影像資料，Discriminator network的工作則是判斷Generator network給的Data為真或假，然後給與「回饋」。Generator network可根據這個「回饋」來Training提高「工藝」，不斷調整Model的各項Parameter，最後使其產生影像的「工藝」精確到令Discriminator network難以分辨，因此完成Training出真正的data分布的model。  
![](https://i.imgur.com/XWqCpfz.png)
圖1-1: GAN流程圖

Discriminator network 的工作(如圖1-2)，簡單的說，就是給予一張圖，相似度越高，表示越像從dataset的model。相反的則為Generator network偽造的。  
 ![](https://i.imgur.com/NeapVM6.png)
圖1-2: Discriminator network模型圖

Generator network(如圖1-3)的角色像是工匠，工作是偽造圖片，跟Discriminator network工作相反，如果Generator network要輸出一張圖片，則要輸入一個隨機數。因此可以想像他是個亂數產生器，輸入許多不同seed產生不同圖片。如果training非常完美，則輸出非常完美圖片。Generator network比較像逆向的CNN model。  

![](https://i.imgur.com/e59VMeM.png)
圖1-3: Generator network模型圖

**1.2 CycleGAN**
CycleGAN突破非配對圖像集之間轉換限制，如斑馬與馬可以CycleGAN轉換，但貓與狗不同動物則無法轉換，因此可以做電腦視覺合成圖特效。Conditional GAN可用來訓練(Training)配對好的dataset，但非配對好的dataset，則無法訓練(Training)。如圖1-4的左圖，有輪廓線條與圖像彼此配對契合，則以Conditional GAN訓練(Training)做轉換較為合適；如圖1-4的右圖，圖片與油畫非匹配欲做轉換，以CycleGAN做訓練(Training)為適合。  
![](https://i.imgur.com/aSHZiIY.png)
  圖1-4: Paired與Unpaired





 
 
 
### 2.Inference’s effect
 本組將Python程式在Server#1 linxu(ubuntu)作業系統上，送入GPN運算，截圖顯示在185/200中，尚有3小時33分鐘完成訓練(Training)，訓練結果送入summer2winter_yosemite目錄夾(如圖2-1)。
 ![](https://i.imgur.com/AIKqCY9.png)
圖2-1: Server#1 Training過程截圖
 

![](https://i.imgur.com/Sjhtc8n.png)
圖2-2是Server#2 linxu(ubuntu)作業系統上，送入GPN運算，截圖顯示在是訓練完成度11%，訓練結果存入Vangogh2photo。
 
![](https://i.imgur.com/xmJb9KW.png)  
圖2-3:apple2orange model
 
 
![](https://i.imgur.com/Gwmt9NU.png)  
圖2-4: Realphoto2Vangogh model


### 3. Compare with other method

本次實驗方法共以下有三種
> - [Super fast color transfer between images](https://github.com/jrosebr1/color_transfer)
> - [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576v2.pdf)
> - [CycleGan](https://arxiv.org/abs/1703.10593)
> 
其中CycleGan為本次主要方法，其餘為baseline方法
#### Compare’s analysis
  ---
<!--   ![](https://i.imgur.com/Or1aQsH.jpg) =>![](https://i.imgur.com/TRwlWgT.jpg) -->

  
  
|Real Photo|Vangogh|
| -------- | -------- |
|![](https://i.imgur.com/KWv6TgG.jpg)|![](https://i.imgur.com/my9bLEs.jpg)

|method| mean and standard deviation   | CNN | cycleGAN|
|--------| -------- | -------- | -------- |
|result|![](https://i.imgur.com/HvtQRTn.png)|![](https://i.imgur.com/FIjUMhk.png)|![](https://i.imgur.com/jwvP0A0.png)


---

| Orange(source) | Apple(target)|
| -------- | -------- |
|  ![](https://i.imgur.com/Or1aQsH.jpg)     | ![](https://i.imgur.com/TRwlWgT.jpg) |

|method| mean and standard deviation   | CNN | cycleGAN|Photoshop|
|--------| -------- | -------- | -------- |--------|
|result|![](https://i.imgur.com/m4sv33w.png)|![](https://i.imgur.com/cA5hufP.png)| ![](https://i.imgur.com/cbXzVUN.png) |![](https://i.imgur.com/VUC5E31.png)
#### pros & cons 
----

1. mean and standard deviation 
圖片的像素以LAB色彩空間儲存，透過其像素強度的平均值與標準差，針對色彩做轉換。[[註1]](https://github.com/jrosebr1/color_transfer)

    優點：
    * 運算快速
    * 清晰度高
    * 運算資源需求低
    
    缺點：
    * 材質幾乎沒改變
    * 顏色幾乎沒改變
2. CNN
使用基礎的卷積神經網路[[註2]](https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py)
    
    優點：
    * 輪廓清晰
    * 表面材質轉換效果出色
    * 光影效果保留度高

    缺點：
    * 圖片佈局要相似，不然轉換效果差[[附圖]]()
    * 轉換時間長
    * 顏色轉換效果差
    * 因不像cycle gan有Discriminator，故轉換出來有時會失真
3. cycleGAN
詳細內容可參見introduction及[範例](https://github.com/aitorzip/PyTorch-CycleGAN)
    優點:
    * 訓練完後不需要target image也能轉換圖片內目標物件
    * 訓練的datasets不需要佈局相似的成對圖片
    * model轉換圖片速度快
    * 顏色轉換效果好且精確
    
    缺點：
    * 訓練時間長
    * 所需運算資源較高
    * 當domain差太多較難訓練
    * 材質轉換效果普通
    * 圖片內物件清晰度下降


Reference
1.CHIEN-YI WANG, 2017, 用GAN來實現更全面的圖像風格轉換 CYCLEGAN: UNPAIRED IMAGE-TO-IMAGE TRANSLATION USING CYCLE-CONSISTENT ADVERSARIAL NETWORKS, https://data-sci.info/2017/05/25/cyclegan/
