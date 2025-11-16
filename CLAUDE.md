## WSIツールボックス

### 基本事項

WSIデータを様々な形で活用する
- 形式を問わず、パッチ分割しhdf5に固める
- 基盤モデルなどに通して、パッチの埋め込みを取得
- クラスタリングおよびクラスタ番号を指定したサブクラスタリングなど包括的な解析を提供

## 開発に関して

- つねに　`uv` を使い、直接 `python` `pip` を使わない
- `cli` は下記のように `pydantic-autocli` を使ってサブコマンドベースのコマンドラインツールとしている

### AutoCLI の使い方

Key patterns:
- `def run_foo_bar(self, args):` → `python script.py foo-bar`
- `def prepare(self, args):` → shared initialization  
- `class FooBarArgs(AutoCLI.CommonArgs):` → command arguments
- Return `True`/`None` (success), `False` (fail), `int` (exit code)

For details: `python your_script.py --help`
