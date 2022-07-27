import os, shutil
from flask import Flask, request, redirect, url_for, render_template, Markup
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from PIL import Image
import numpy as np

UPLOAD_FOLDER = "./static/images/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

labels =  [
  'リンゴ',  # 0：りんご
  '観賞魚',  # 1：観賞魚
  '赤ちゃん',  # 2：赤ちゃん
  'クマ',  # 3：クマ
  'ビーバー',  # 4：ビーバー
  'ベッド',  # 5：ベッド
  '蜂',  # 6：蜂
  'カブトムシ',  # 7：カブトムシ
  '自転車',  # 8：自転車
  'ボトル',  # 9：ボトル
  'ボウル',  # 10：ボウル
  '少年',  # 11：少年
  '橋',  # 12：橋
  'バス',  # 13：バス
  '蝶',  # 14：蝶
  'ラクダ',  # 15：ラクダ
  '缶',  # 16：缶
  '城',  # 17：城
  '毛虫',  # 18：毛虫
  '牛',  # 19：牛
  '椅子',  # 20：椅子
  'チンパンジー',  # 21：チンパンジー
  '時計',  # 22：時計
  '雲',  # 23：雲
  'ゴキブリ',  # 24：ゴキブリ
  'ソファー',  # 25：ソファー
  'カニ',  # 26：カニ
  'ワニ',  # 27：ワニ
  'カップ',  # 28：カップ
  '恐竜',  # 29：恐竜
  'イルカ',  # 30：イルカ
  '象',  # 31：象
  'ヒラメ',  # 32：ヒラメ
  '森',  # 33：森
  'キツネ',  # 34：キツネ
  '少女',  # 35：少女
  'ハムスター',  # 36：ハムスター
  '家',  # 37：家
  'カンガルー',  # 38：カンガルー
  'コンピューターのキーボード',  # 39：コンピューターのキーボード
  'ランプ',  # 40：ランプ
  '芝刈り機',  # 41：芝刈り機
  'ヒョウ',  # 42：ヒョウ
  'ライオン',  # 43：ライオン
  'トカゲ',  # 44：トカゲ
  'ロブスター',  # 45：ロブスター
  '成人男性',  # 46：成人男性
  'もみじ',  # 47：もみじ
  'オートバイ',  # 48：オートバイ
  '山',  # 49：山
  'ねずみ',  # 50：ねずみ
  'きのこ',  # 51：きのこ
  'オーク',  # 52：オーク
  'オレンジ',  # 53：オレンジ
  '蘭',  # 54：蘭
  'カワウソ',  # 55：カワウソ
  'ヤシ',  # 56：ヤシ
  '洋ナシ',  # 57：洋ナシ
  'ピックアップトラック',  # 58：ピックアップトラック
  '松',  # 59：松
  '平野',  # 60：平野
  '皿',  # 61：皿
  'ポピー',  # 62：ポピー
  'ヤマアラシ',  # 63：ヤマアラシ
  'フクロネズミ',  # 64：フクロネズミ
  'ウサギ',  # 65：ウサギ
  'アライグマ',  # 66：アライグマ
  'エイ',  # 67：エイ
  '道路',  # 68：道路
  'ロケット',  # 69：ロケット
  'バラ',  # 70：バラ
  '海',  # 71：海
  'アザラシ',  # 72：アザラシ
  'サメ',  # 73：サメ
  'トガリネズミ',  # 74：トガリネズミ
  'スカンク',  # 75：スカンク
  '超高層ビル',  # 76：超高層ビル
  'カタツムリ',  # 77：カタツムリ
  'ヘビ',  # 78：ヘビ
  'クモ',  # 79：クモ
  'リス',  # 80：リス
  '路面電車',  # 81：路面電車
  'ひまわり',  # 82：ひまわり
  'パプリカ',  # 83：パプリカ
  'テーブル',  # 84：テーブル
  'タンク',  # 85：タンク
  '電話',  # 86：電話
  'テレビ',  # 87：テレビ
  'トラ',  # 88：トラ
  'トラクター',  # 89：トラクター
  '電車',  # 90：電車
  'マス',  # 91：マス
  'チューリップ',  # 92：チューリップ
  'カメ',  # 93：カメ
  'ワードローブ',  # 94：ワードローブ
  'クジラ',  # 95：クジラ
  '柳',  # 96：柳
  'オオカミ',  # 97：オオカミ
  '成人女性',  # 98：成人女性
  'ミミズ',  # 99：ミミズ
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
            result += "<p>" + str(round(ratio*100, 1)) + "%の確率で" + label + "です。</p>"
            
        return render_template("result.html", result=Markup(result), filepath=filepath)
    else:
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)