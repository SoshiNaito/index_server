# 自作検索エンジン
専門学校の卒業制作で作成した検索エンジンです！
開発規模：6人
担当:PM 検索サーバー
こだわった点：
一般的な検索エンジンの検索方法として単語を連ねて検索するという方法があります。
例）東京でお昼ご飯が食べたい時 → 東京 ご飯
この検索方法は検索したい文章などから特徴的な部分を抽出しそれを名詞化して、検索するというプロセスが必要です。
この方法は一見簡単なことですが、ネットリテラシーがそこまで高くないお年寄りの方や子供、また専門的な文章だと専門外の人にとっては難しいのではと考えました。
そこで、こう言った問題を解決すべく作成したのが文章検索に特化した自作検索エンジンです。
この検索エンジンは自然言語処理モデルのBERTを利用することで文章検索に特化した検索をすることができます。

中身：
1.検索対象の文章を先にモデルに通し文ベクトルというベクトル表現に変換し、準備しておく。
2.入力された文章をモデルに通し文ベクトルに変換する。
3.文ベクトルどおしをコサイン類似度を使い比較。
4.類似度順に検索結果として表示する。

苦労した点：
自分以外が授業以外で開発未経験だったため、周りのモチベーションをあげることが大変でした。
相手がこなせる粒度でタスクを渡したり、やることで今後どう言ったことができるようになるか等を意識しました。
また自然言語処理についてもゼロからの勉強だったため、キャッチアップが大変でした。
同キャンパス内にある大学の研究室にお願いして、一緒に勉強させていただいたり、英語の資料等を読み漁りなんとか実装にこぎつけました。


DEMO:
https://twitter.com/7110It/status/1234199528422563840?s=20

### 使用技術
クライアント：React Sass
サーバー：Go Python BERT

# 検索サーバー
1. Pythonのバージョンを3.7.4にする
2. `pip install -r requirements.txt`をコマンドで実行
3. 上が完了したら、app.pyを開くとサーバーが起動する
