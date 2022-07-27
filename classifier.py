import os, shutil
from flask import Flask, request, redirect, url_for, render_template, Markup
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from PIL import Image
import numpy as np

UPLOAD_FOLDER = "./static/images/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

labels =  [
  'apples',  # 0：りんご
  'aquarium fish',  # 1：観賞魚
  'baby',  # 2：赤ちゃん
  'bear',  # 3：クマ
  'beaver',  # 4：ビーバー
  'bed',  # 5：ベッド
  'bee',  # 6：蜂
  'beetle',  # 7：カブトムシ
  'bicycle',  # 8：自転車
  'bottles',  # 9：ボトル
  'bowls',  # 10：ボウル
  'boy',  # 11：少年
  'bridge',  # 12：橋
  'bus',  # 13：バス
  'butterfly',  # 14：蝶
  'camel',  # 15：ラクダ
  'cans',  # 16：缶
  'castle',  # 17：城
  'caterpillar',  # 18：毛虫
  'cattle',  # 19：牛
  'chair',  # 20：椅子
  'chimpanzee',  # 21：チンパンジー
  'clock',  # 22：時計
  'cloud',  # 23：雲
  'cockroach',  # 24：ゴキブリ
  'couch',  # 25：ソファー
  'crab',  # 26：カニ
  'crocodile',  # 27：ワニ
  'cups',  # 28：カップ
  'dinosaur',  # 29：恐竜
  'dolphin',  # 30：イルカ
  'elephant',  # 31：象
  'flatfish',  # 32：ヒラメ
  'forest',  # 33：森
  'fox',  # 34：キツネ
  'girl',  # 35：少女
  'hamster',  # 36：ハムスター
  'house',  # 37：家
  'kangaroo',  # 38：カンガルー
  'computer keyboard',  # 39：コンピューターのキーボード
  'lamp',  # 40：ランプ
  'lawn-mower',  # 41：芝刈り機
  'leopard',  # 42：ヒョウ
  'lion',  # 43：ライオン
  'lizard',  # 44：トカゲ
  'lobster',  # 45：ロブスター
  'man',  # 46：成人男性
  'maple',  # 47：もみじ
  'motorcycle',  # 48：オートバイ
  'mountain',  # 49：山
  'mouse',  # 50：ねずみ
  'mushrooms',  # 51：きのこ
  'oak',  # 52：オーク
  'oranges',  # 53：オレンジ
  'orchids',  # 54：蘭
  'otter',  # 55：カワウソ
  'palm',  # 56：ヤシ
  'pears',  # 57：洋ナシ
  'pickup truck',  # 58：ピックアップトラック
  'pine',  # 59：松
  'plain',  # 60：平野
  'plates',  # 61：皿
  'poppies',  # 62：ポピー
  'porcupine',  # 63：ヤマアラシ
  'possum',  # 64：フクロネズミ
  'rabbit',  # 65：ウサギ
  'raccoon',  # 66：アライグマ
  'ray',  # 67：エイ
  'road',  # 68：道路
  'rocket',  # 69：ロケット
  'roses',  # 70：バラ
  'sea',  # 71：海
  'seal',  # 72：アザラシ
  'shark',  # 73：サメ
  'shrew',  # 74：トガリネズミ
  'skunk',  # 75：スカンク
  'skyscraper',  # 76：超高層ビル
  'snail',  # 77：カタツムリ
  'snake',  # 78：ヘビ
  'spider',  # 79：クモ
  'squirrel',  # 80：リス
  'streetcar',  # 81：路面電車
  'sunflowers',  # 82：ひまわり
  'sweet peppers',  # 83：パプリカ
  'table',  # 84：テーブル
  'tank',  # 85：タンク
  'telephone',  # 86：電話
  'television',  # 87：テレビ
  'tiger',  # 88：トラ
  'tractor',  # 89：トラクター
  'train',  # 90：電車
  'trout',  # 91：マス
  'tulips',  # 92：チューリップ
  'turtle',  # 93：カメ
  'wardrobe',  # 94：ワードローブ
  'whale',  # 95：クジラ
  'willow',  # 96：柳
  'wolf',  # 97：オオカミ
  'woman',  # 98：成人女性
  'worm',  # 99：ミミズ
]
n_class = len(labels)
img_size = 32
n_result = 5  # 上位3つの結果を表示

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/result", methods=["GET","POST"])
def result():
    if request.method == "POST":
        # ファイルの存在と形式を確認
        if "file" not in request.files:
            print("File doesn't exist!")
            return redirect(url_for("index"))
        file = request.files["file"]
        if not allowed_file(file.filename):
            print(file.filename + ": File not allowed!")
            return redirect(url_for("index"))

        # ファイルの保存
        if os.path.isdir(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER) 
        os.mkdir(UPLOAD_FOLDER)
        filename = secure_filename(file.filename)  # ファイル名を安全なものに
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # 画像の読み込み
        image = Image.open(filepath)
        image = image.convert("RGB")
        image = image.resize((img_size, img_size))
        x = np.array(image, dtype=float)
        x = x.reshape(1, img_size, img_size, 3) / 255        

        # 予測
        model = load_model("./model_and_weight.h5")
        y = model.predict(x)[0]
        sorted_idx = np.argsort(y)[::-1]  # 降順でソート
        result = ""
        for i in range(n_result):
            idx = sorted_idx[i]
            ratio = y[idx]
            label = labels[idx]
            # result += "<p>" + str(round(ratio*100, 1)) + "%の確率で" + label + "です。</p>"
            result += "<p>"+ label +"</p>"
        return render_template("result.html", result=Markup(result), filepath=filepath)
    else:
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)