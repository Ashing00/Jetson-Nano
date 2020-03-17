#===========================================
#convert yolov3 to tensorRT and run on Jetson Nano
#===========================================
#----------------------------
#準備環境
#----------------------------
#A. check Tensorrt 已經裝在Jetson Nano
$ dpkg -l | grep TensorRT
#有出現東西就是有，順便可以看一下版本
#預設Jetpack image就會有裝,所以不用重裝
#我的Jetpack image 是4.2 版,cuda=10.0 ,TensorRT=5.0.6.3

#B. 安裝onnx
$sudo apt install protobuf-compiler libprotoc-dev
$pip install onnx==1.4.1

#C. 安裝tensorRT for python3
$pip list 
#應該要找到tensorrt  5.0.6.3
#但是由於我們通常建立了一個虛擬python環境.所以如果你用
#workon AI 進python虛擬環境,那麼系統其實會找不到tensorrt x.x.x.x
#所以要在AI python虛擬環境,裝tensorrt
#最簡單的方式如下
#找到系統目錄/usr/lib/python3.6/dist-packages/
#copy  目錄 
/usr/lib/python3.6/dist-packages/tensorrt
/usr/lib/python3.6/dist-packages/tensort-5.0.6.3.dist-info
/usr/lib/python3.6/dist-packages/graphsurgeon
/usr/lib/python3.6/dist-packages/graphsurgeon-0.3.2.dist-info
/usr/lib/python3.6/dist-packages/uff
/usr/lib/python3.6/dist-packages/uff-0.5.5.dist-info
 
#到  ~/.virtualenvs/AI/lib/python3.6/site-packages
#AI  是你所建立的虛擬環境名稱
#再次驗證 tensorrt x.x.x.x
$pip list 

#----------------------------
#下載source code
#----------------------------
#1.download 底下github source code
git clone https://github.com/jkjung-avt/tensorrt_demos

#2.安裝pycuda
$ cd ${HOME}/project/tensorrt_demos/ssd
$ ./install_pycuda.sh
==>也可以直接 pip install pycuda
#3下載Yolov3 原始權重
$ cd ${HOME}/project/tensorrt_demos/yolov3_onnx
$ ./download_yolov3.sh

#4轉換yolov3-416 model to onnx,轉完會產生yolov3-416.onnx ,這一步大概要跑20-30分鐘,不過這一步可以先在可以先在PC端完成比較快
$ python3 yolov3_to_onnx.py --model yolov3-416

#5轉換yolov3-416.onnx to yolov3-416.onnx.trt ,#這一步大概也要跑20-30分鐘,這一步一定要在Jetson Nano 上轉
$ python3 onnx_to_tensorrt.py --model yolov3-416

#6下載測試圖片
$ wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg -O ${HOME}/Pictures/dog.jpg
#7跑測試圖片
$ python3 trt_yolov3.py --model yolov3-416 --image --filename ${HOME}/Pictures/dog.jpg

#8 或是打開USB Camera live show
python3 trt_yolov3.py --model yolov3-416 --usb --vid 0 --height 720 --width 1280

PS. for  open  USB  camera 先修改底下設定
~\tensorrt_demos\utils\camera.py
USB_GSTREAMER = False		# True to False 


#-------------------------------------------------------
#如果你有訓練自己的自己的yolov3 model  
#須修改下列事項修改下列事項
#------------------------------------------------------
1.for  open USB camera
~\tensorrt_demos\utils\camera.py
USB_GSTREAMER = False		#ashing True to False 

2.motify class label
~\tensorrt_demos\utils\yolov3_classes.py
3.motify class number
~\tensorrt_demos\utils\yolov3.py
    def __init__(self,
                 yolo_masks,
                 yolo_anchors,
                 nms_threshold,
                 yolo_input_resolution,
                 category_num=10):		#ashing 80=>10
				 ##...........................
	        if 'tiny' in model:
            self.output_shapes = [(1, 45, h // 32, w // 32),
                                  (1, 45, h // 16, w // 16)]
        else:
            self.output_shapes = [(1, 45, h // 32, w // 32),		#ashing 255=>45  (10+5)*3 ; 10個類別
                                  (1, 45, h // 16, w // 16),
                                  (1, 45, h //  8, w //  8)]			 


#~\tensorrt_demos\yolov3_onnx\yolov3_to_onnx.py
	my_dim=45		#ashing	 45=(10+5)*3   replace	 (80+5)*3=255
	if 'tiny' in args.model:
		output_tensor_dims['016_convolutional'] = [my_dim, d // 32, d // 32]
		output_tensor_dims['023_convolutional'] = [my_dim, d // 16, d // 16]
	else:
		output_tensor_dims['082_convolutional'] = [my_dim, d // 32, d // 32]
		output_tensor_dims['094_convolutional'] = [my_dim, d // 16, d // 16]
		output_tensor_dims['106_convolutional'] = [my_dim, d //	 8, d //  8]

##重新轉換後再run 
python3 trt_yolov3.py --model yolov3-416 --usb --vid 0 --height 720 --width 1280


#-------------------------------------------------------
#For PC 的環境
#------------------------------------------------------
可以直接參考底下連結.deb 跟.tar 兩種方式都要run
https://blog.csdn.net/shwan_ma/article/details/103637739


