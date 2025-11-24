from typing import Iterable, Optional, TypeVar

from tqdm import tqdm

T = TypeVar("T")


class BaseProgress:
    """Base interface for progress bars"""

    def update(self, n: int = 1) -> None:
        raise NotImplementedError

    def set_description(self, desc: str = None, refresh: bool = True) -> None:
        raise NotImplementedError

    def set_postfix(self, ordered_dict=None, **kwargs) -> None:
        raise NotImplementedError

    def refresh(self) -> None:
        """Force refresh the progress bar display"""
        pass

    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TqdmProgress(BaseProgress):
    """tqdm wrapper"""

    def __init__(self, iterable: Optional[Iterable[T]] = None, total: Optional[int] = None, desc: str = "", **kwargs):
        self._pbar = tqdm(iterable=iterable, total=total, desc=desc, **kwargs)

    def update(self, n: int = 1) -> None:
        self._pbar.update(n)

    def set_description(self, desc: str = None, refresh: bool = True) -> None:
        self._pbar.set_description(desc, refresh=refresh)

    def set_postfix(self, ordered_dict=None, **kwargs) -> None:
        self._pbar.set_postfix(ordered_dict, **kwargs)

    def refresh(self) -> None:
        self._pbar.refresh()

    def close(self) -> None:
        self._pbar.close()

    def __iter__(self):
        return iter(self._pbar)


class StreamlitProgress(BaseProgress):
    """Streamlit progress bar wrapper"""

    def __init__(self, iterable: Optional[Iterable[T]] = None, total: Optional[int] = None, desc: str = "", **kwargs):
        # import streamlit as st はここに置くこと
        # import streamlit as st はここに置くこと
        # import streamlit as st はここに置くこと
        import streamlit as st  # noqa: E402

        self.iterable = iterable
        self.total = (
            total
            if total is not None
            else (len(iterable) if iterable is not None and hasattr(iterable, "__len__") else None)
        )
        self.desc = desc
        self.n = 0
        self.kwargs = kwargs

        # 説明テキスト用のコンテナ
        self.text_container = st.empty()
        if desc:
            self.text_container.text(desc)
        # プログレスバー
        self.progress_bar = st.progress(0)
        # 後置テキスト用のコンテナ
        self.postfix_container = st.empty()

    def update(self, n: int = 1) -> None:
        """進捗を更新する"""
        self.n += n
        if self.total:
            self.progress_bar.progress(min(self.n / self.total, 1.0))

    def set_description(self, desc: str = None, refresh: bool = True) -> None:
        """説明テキストを更新する"""
        if desc is not None:
            self.desc = desc
            # self.text_container.text(desc)
            self.text_container.markdown(
                '<p style="font-size:14px; color:gray;">' + desc + "</p>", unsafe_allow_html=True
            )

    def set_postfix(self, ordered_dict=None, **kwargs) -> None:
        """後置テキストを設定する"""
        # ordered_dictとkwargsを組み合わせる
        postfix_dict = {}
        if ordered_dict:
            postfix_dict.update(ordered_dict)
        if kwargs:
            postfix_dict.update(kwargs)

        if postfix_dict:
            # 辞書を文字列に変換して表示
            postfix_str = ", ".join(f"{k}={v}" for k, v in postfix_dict.items())
            self.postfix_container.text(f"状態: {postfix_str}")

    def close(self) -> None:
        """プログレスバーを完了状態にする"""
        if self.total:
            self.progress_bar.progress(1.0)
        self.text_container.empty()

    def refresh(self):
        """不要なので何もしない"""
        pass

    def __iter__(self):
        """イテレータとして使用できるようにする"""
        if self.iterable is None:
            raise ValueError("このプログレスバーはイテレータとして使用できません")

        for obj in self.iterable:
            yield obj
            self.update(1)

        self.close()

    def __enter__(self):
        """コンテキストマネージャとして使用できるようにする"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキスト終了時に呼ばれる"""
        self.close()


class DummyProgress(BaseProgress):
    """Dummy progress bar (no output)"""

    def __init__(self, iterable: Optional[Iterable[T]] = None, total: Optional[int] = None, desc: str = "", **kwargs):
        self.iterable = iterable
        self.total = total
        self.desc = desc
        self.n = 0

    def update(self, n: int = 1) -> None:
        self.n += n

    def set_description(self, desc: str = None, refresh: bool = True) -> None:
        if desc is not None:
            self.desc = desc

    def set_postfix(self, ordered_dict=None, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass

    def __iter__(self):
        if self.iterable is None:
            raise ValueError("No iterable provided")
        for obj in self.iterable:
            yield obj
            self.update(1)


def Progress(
    iterable: Optional[Iterable[T]] = None,
    backend: str = "tqdm",
    total: Optional[int] = None,
    desc: str = "",
    **kwargs,
) -> BaseProgress:
    """
    Create a progress bar with the specified backend

    Args:
        iterable: Optional iterable to track
        backend: Backend type ("tqdm", "streamlit", "dummy")
        total: Total iterations (required if iterable is None)
        desc: Description text
        **kwargs: Additional arguments passed to the backend

    Returns:
        BaseProgress instance
    """
    if backend == "tqdm":
        return TqdmProgress(iterable=iterable, total=total, desc=desc, **kwargs)
    elif backend == "streamlit":
        try:
            return StreamlitProgress(iterable=iterable, total=total, desc=desc, **kwargs)
        except ImportError:
            print("streamlit not found, falling back to dummy progress")
            return DummyProgress(iterable=iterable, total=total, desc=desc, **kwargs)
    elif backend == "dummy":
        return DummyProgress(iterable=iterable, total=total, desc=desc, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
