# 検索サーバー

1. Pythonのバージョンを3.7.4にする
2. `pip install -r requirements.txt`をコマンドで実行
3. 上が完了したら、app.pyを開く
4. `curl -X POST -H "Content-Type: application/json" -d '{"q":"sensuikan1973"}' localhost:8080/`を実行する
5. 200番だと成功
