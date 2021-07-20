##画像を学習用とテスト用に分ける
from PIL import Image
import os, glob, shutil
import numpy as np
from sklearn import model_selection


def generate():
    face_classes = ["MyFace", "OtherFace1"]
    num_face_classes = len(face_classes) #要素数

    use_count = int(input("使用データ数："))
    test_data_count = int(use_count*0.3) #使用データ数の3割をテストデータに使用

    x_train = [] #テスト用のデータを格納するリスト
    x_test = []
    y_train = [] #学習用のデータを格納するリスト
    y_test = []

    for index, classlabel in enumerate(face_classes): #リストのインデックスと要素を代入（zip()と変わらない）
        photo_dir = "./" + classlabel #画像のディレクトリ名
        files = glob.glob(photo_dir + "/*.jpg") #画像を一枚ずつ取得（glob()ではワイルドカード使用可能）

        for i, file in enumerate(files):
            if i >= use_count: #エラー処理
                break

            image_open = Image.open(file) #画像を開く
            image_rgb = image_open.convert("RGB") #RGB化
            image = image_rgb.resize((50, 50))
            data = np.asarray(image) #画像を行列化（コピーを作成しない）

            if i < test_data_count: #枚数がtest_data_countと一致しないなら
                x_test.append(data)
                y_test.append(index)
            else:
                for face_angle in range(-5, 5, 5):
                    img_roll = image.rotate(face_angle) #-5度から5度まで5度ずつ回転
                    data = np.asarray(img_roll) #回転した画像を数値化
                    x_train.append(data)
                    y_train.append(index)

                    img_rurn1 = img_roll.transpose(Image.FLIP_LEFT_RIGHT) #左右反転
                    data = np.asarray(img_rurn1) #数値に変換
                    x_train.append(data)
                    y_train.append(index)

    x_train = np.array(x_train) #TensorFlowで扱いやすいようにnumpy形式に変換
    x_test =  np.array(x_test) #np.array()はデフォルトでcopy=Trueとなっているため自動的にコピーが作成される
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    xy_box = (x_train, x_test, y_train, y_test)

    save_dir = "Result"
    if os.path.exists(f"./{save_dir}") == True:
        shutil.rmtree(f"./{save_dir}")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    np.save(f"./{save_dir}/face_data.npy", xy_box)


if __name__ == "__main__":
    generate()
