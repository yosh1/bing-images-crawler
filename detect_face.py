import glob
import os
import cv2

names = ['ota','tanaka']
out_dir = "./data/out/"
os.makedirs(out_dir, exist_ok=True)
for i in range(len(names)):

    in_dir = "./data/in/"+names[i]+"/*.jpg"
    in_jpg = glob.glob(in_dir)
    os.makedirs(out_dir + names[i], exist_ok=True)
    # print(in_jpg)
    print(len(in_jpg))
    for num in range(len(in_jpg)):
        image=cv2.imread(str(in_jpg[num]))
        if image is None:
            print("Not open:",num)
            continue

        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")


        face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))

        if len(face_list) > 0:
            for rect in face_list:
                x,y,width,height=rect
                image = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
                if image.shape[0]<64:
                    continue
                image = cv2.resize(image,(64,64))

                fileName=os.path.join(out_dir + names[i],str(num)+".jpg")
                cv2.imwrite(str(fileName),image)
                print(str(num)+".jpgを保存しました.")

        else:
            print("no face")
            continue
        print(image.shape)
