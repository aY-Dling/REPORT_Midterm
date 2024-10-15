# google-colab平台訓練模型案例
南華大學_跨領域-人工智慧_MidReport

11124208王品雯、11124209蔡岱伶
## 數據集介紹
![截圖01](https://github.com/aY-Dling/REPORT_Midterm/blob/main/%E6%88%AA%E5%9C%9601.png?raw=true)
数据集来自Kaggle，质量很高，由知名医院的专业人员严格审核标注，如图所示数据有4种类别：
  CNV：具有新生血管膜和相关视网膜下液的脉络膜新血管形成
  DME：糖尿病性黄斑水肿与视网膜增厚相关的视网膜内液
  DRUSEN:早期AMD中存在多个玻璃疣
  NORMAL：视网膜正常，没有任何视网膜液或水肿
  ![截圖02](https://github.com/aY-Dling/REPORT_Midterm/blob/main/%E6%88%AA%E5%9C%9602.png?raw=true)
文件大小约为5GB，8万多张图像，分为训练，测试，验证三个文件夹，每个文件夹按照种类不同分成4个子文件夹，其次是具体图像文件。

##數據集下載
挂载文件夹：

from google.colab import drive
drive.mount('/content/gdrive/')
按照提示进行验证，结果如下：
![截圖03](https://github.com/aY-Dling/REPORT_Midterm/blob/main/%E6%88%AA%E5%9C%9603.png?raw=true)
kaggle数据下载：

创建kaggle账户并下载kaggle.json文件。创建账户这里就不介绍了，创建完账户后在“我的账户”-“API”中选择“CREATE NEW API TOKEN”，然后下载kaggle.json文件。

创建kaggle文件夹：

!mkdir -p ~/.kaggle
将kaggle.json文件夹复制到指定文件夹：

!cp /content/gdrive/My\ Drive/kaggle.json ~/.kaggle/
测试是否成功：

!kaggle competitions list
![截圖04](https://github.com/aY-Dling/REPORT_Midterm/blob/main/%E6%88%AA%E5%9C%9604.png?raw=true)
