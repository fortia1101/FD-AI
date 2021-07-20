##畳み込みニューラルネットワークによる画像の判定
import keras
from keras.models import Sequential #ニューラルネットワークのモデルを定義する
from keras.layers import Conv2D #畳み込み演算
from keras.layers import MaxPooling2D #プーリング処理
from keras.layers import Activation #活性化関数
from keras.layers import Dropout #ドロップアウト処理
from keras.layers import Flatten #データを1次元に変換
from keras.layers import Dense #全結合
from keras.utils import np_utils #One-Hotベクトル化
import numpy as np


face_classes = ["MyFace", "OtherFace1"]
num_face_classes = len(face_classes)
image_size = 50


##学習用の関数
def train_model(x, y):
    #---------------------------------------------------------------------------

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding="same", input_shape=x.shape[1:]))
    model.add(Activation("relu")) #負の値を除去して活性化関数を作成

    #---------------------------------------------------------------------------

    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2))) #2×2の範囲で最大値を取得
    model.add(Dropout(rate=0.25)) #ランダムでニューロンを無効化することによる過学習の抑制（25％抑制）
    #偏ったニューロンに依存しないため

    #---------------------------------------------------------------------------

    model.add(Conv2D(64, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))

    #---------------------------------------------------------------------------

    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(Activation("rule"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #---------------------------------------------------------------------------

    model.add(Flatten())
    model.add(Dence(512))
    model.add(Actiivation("rule"))
    model.add(Dropout(0.5))
    model.add(Dense(num_face_classes))
    model.add(Activation("softmax"))

    #---------------------------------------------------------------------------

    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6) #オプティマイザーの作成

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]) #コンパイル

    model.fit(x, y, batch_size=32, epochs=20) #学習

    model.save("./Result/face_cnn.h5") #モデルの保存
    return model


##テスト用の関数
def evaluate_model(model, x, y):
    scores = model.evaluate(x, y, verbose=1)
    print("Test Loss:", scores[0])
    print("Test Accuracy:", scores[1])


##実行
def cnn_main():
    #.npy形式のファイルからデータを取り出す
    x_train, x_test, y_train, y_test = np.load("./Result/face_data.npy", allow_pickle=True) #allow_pickleはnpyファイルとして保存されているpickleオブジェクトを読み込むかどうかを指定する

    x_train = x_train.astype("float") / 256 #データの正規化：ndarrayの各要素のデータ型をfloatに変換したのち256で割る
    x_test = x_test.astype("float") / 256

    y_train = np_utils.to_categorical(y_train, num_face_classes) #One-Hotベクトル化
    y_test = np_utils.to_categorical(y_test, num_face_classes)

    model = train_model(x_train, y_train)
    evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    cnn_main()
