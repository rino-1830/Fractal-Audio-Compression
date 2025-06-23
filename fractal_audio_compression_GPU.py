# -*- coding: utf-8 -*-
"""
ステレオ音源の時間波形に対する簡易フラクタル圧縮の実装。

このスクリプトではブロック分割した音源の自己相似性を利用して、
線形変換パラメータのみを保存する簡易的なフラクタル圧縮を行う。
"""

# GPU計算を行うためCuPyを使用
import cupy as cp
import numpy as np  # ファイル入出力のためにNumPyも使用
from scipy.io import wavfile
from tqdm import tqdm

# cp.errstate が存在しない場合に備えて代替を定義
try:
    cp.errstate  # type: ignore[attr-defined]
    errstate = cp.errstate
except AttributeError:  # 古いCuPyでは errstate がない
    from contextlib import contextmanager

    @contextmanager
    def errstate(**kwargs):
        """NumPy の errstate と互換の簡易実装"""
        if hasattr(cp, "seterr"):
            old = cp.seterr(**kwargs)
        else:
            old = None
        try:
            yield
        finally:
            if old is not None:
                cp.seterr(**old)


def _prepare_domains(audio: cp.ndarray, block_size: int, search_step: int):
    """domainブロックと統計量をGPU上で計算しておく"""
    length = int(audio.shape[0])
    starts = cp.arange(0, length - 2 * block_size + 1, search_step, dtype=cp.int32)
    # インデックスをまとめて生成してdomainブロックを取得
    idx = starts[:, None] + cp.arange(0, 2 * block_size, 2, dtype=cp.int32)
    domains = audio[idx].astype(cp.float32)
    means = cp.mean(domains, axis=1, dtype=cp.float32)
    centered = domains - means[:, None]
    variances = cp.sum(centered**2, axis=1, dtype=cp.float32)
    return domains, centered, means, variances, starts


def compress(audio: cp.ndarray, block_size: int = 1024, search_step: int = 512):
    """音源をGPU上でフラクタル圧縮する"""
    orig_length = audio.shape[0]
    # ブロック単位にそろえるためゼロ埋めを行う
    pad = (-orig_length) % block_size
    if pad:
        audio = cp.pad(audio, (0, pad))
    length = audio.shape[0]
    # domainブロックと統計量をまとめて準備
    domains, centered, means, variances, starts = _prepare_domains(
        audio, block_size, search_step
    )
    num_blocks = int(length) // block_size
    best_starts = cp.empty(num_blocks, dtype=cp.int32)
    best_scales = cp.empty(num_blocks, dtype=cp.float32)
    best_offsets = cp.empty(num_blocks, dtype=cp.float32)
    # 残り予測時間を表示するため tqdm オブジェクトを変数として保持
    pbar = tqdm(range(0, int(length), block_size), desc="compress")
    # 音源をrangeブロックに分割して処理
    for i, start in enumerate(pbar):
        range_block = audio[start : start + block_size].astype(cp.float32)
        r_mean = cp.mean(range_block, dtype=cp.float32)
        y = range_block - r_mean
        # 全domainブロックに対する線形回帰をまとめて計算
        # 分散が0の場合は係数を0とし、警告を抑制する
        # CuPy に errstate がない環境でも動作するよう独自実装を利用
        with errstate(divide="ignore", invalid="ignore"):
            s = cp.divide(
                centered @ y,
                variances,
                out=cp.zeros_like(variances, dtype=cp.float32),
                where=variances != 0,
            )
        o = r_mean - s * means
        approx = s[:, None] * domains + o[:, None]
        errs = cp.mean((range_block - approx) ** 2, axis=1, dtype=cp.float32)
        idx = cp.argmin(errs)
        best_starts[i] = starts[idx]
        best_scales[i] = s[idx]
        best_offsets[i] = o[idx]
        # tqdm が計算した残り時間をフォーマットして表示
        remaining = pbar.format_dict.get("remaining")
        if remaining is not None:
            eta = pbar.format_interval(remaining)
            pbar.set_postfix_str(f"残り予測時間: {eta}")
    transforms = [
        (int(bs), float(sc), float(ofs))
        for bs, sc, ofs in zip(
            cp.asnumpy(best_starts), cp.asnumpy(best_scales), cp.asnumpy(best_offsets)
        )
    ]
    return {
        "length": int(length),
        "orig_length": int(orig_length),
        "block_size": block_size,
        "search_step": search_step,
        "transforms": transforms,
    }


def _prepare_decompress(params, scale: int = 1):
    """復元で使用するインデックスをGPU上で生成する"""
    block_size = params["block_size"]
    transforms = params["transforms"]
    d_starts = cp.asarray([t[0] for t in transforms], dtype=cp.int32) * scale
    scales = cp.asarray([t[1] for t in transforms], dtype=cp.float32)
    offsets = cp.asarray([t[2] for t in transforms], dtype=cp.float32)
    r_starts = cp.arange(0, params["length"], block_size, dtype=cp.int32) * scale
    # domainブロックの長さを倍率に合わせて拡大し、2サンプルおきに参照
    domain_idx = d_starts[:, None] + cp.arange(
        0, 2 * block_size * scale, 2, dtype=cp.int32
    )
    range_idx = r_starts[:, None] + cp.arange(block_size * scale, dtype=cp.int32)
    return domain_idx, range_idx, scales, offsets


def decompress(params, iterations: int = 8, scale: int = 1):
    """圧縮パラメータからGPU上で音源を復元する"""
    length = params["length"] * scale
    orig_length = params.get("orig_length", params["length"]) * scale
    domain_idx, range_idx, scales, offsets = _prepare_decompress(params, scale)
    # 初期値としてゼロ波形を用意
    audio = cp.zeros(length, dtype=cp.float32)
    new_audio = cp.zeros_like(audio)
    # 進捗から残り時間を得るため tqdm オブジェクトを保持
    pbar = tqdm(range(iterations), desc="decompress")
    for _ in pbar:
        domains = audio[domain_idx]
        approx = scales[:, None] * domains + offsets[:, None]
        new_audio[:] = audio
        # まとめて代入することで高速化
        new_audio[range_idx] = approx
        audio, new_audio = new_audio, audio
        # tqdm の remaining をフォーマットして残り予測時間を表示
        remaining = pbar.format_dict.get("remaining")
        if remaining is not None:
            eta = pbar.format_interval(remaining)
            pbar.set_postfix_str(f"残り予測時間: {eta}")
    # パディングしていた場合は元の長さに切り詰める
    return audio[:orig_length]


def load_wav(path: str):
    """WAVファイルを読み込みGPU用配列に変換する"""
    # 単一チャンネルの場合でもステレオに変換
    rate, data = wavfile.read(path)
    if data.ndim == 1:
        data = np.stack([data, data], axis=-1)
    return rate, cp.asarray(data.astype(np.float32))


def save_wav(path: str, rate: int, data: cp.ndarray):
    """GPU上のデータをWAVとして保存する"""
    # NaNやinfを除去して16bit PCMで保存する
    cpu_data = cp.asnumpy(data)
    safe = np.nan_to_num(cpu_data, nan=0.0, posinf=32767, neginf=-32768)
    clipped = np.clip(safe, -32768, 32767).astype(np.int16)
    wavfile.write(path, rate, clipped)


def calc_mse(original: cp.ndarray, restored: cp.ndarray) -> float:
    """元音源と復元音源の平均二乗誤差を計算する"""
    min_len = min(original.shape[0], restored.shape[0])
    diff = original[:min_len] - restored[:min_len]
    return float(cp.mean(diff**2, dtype=cp.float32).get())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ステレオ音源フラクタル圧縮デモ")
    parser.add_argument("input", help="入力WAVファイル")
    parser.add_argument("output", help="出力WAVファイル")
    parser.add_argument(
        "--mode",
        choices=["compress", "decompress", "evaluate"],
        default="compress",
    )
    parser.add_argument(
        "--params", help="パラメータ保存/読込ファイル", default="params.npz"
    )
    # 圧縮・復元の調整パラメータ
    parser.add_argument("--block-size", type=int, default=1024, help="ブロックサイズ")
    parser.add_argument(
        "--search-step", type=int, default=512, help="domain探索のステップ"
    )
    parser.add_argument("--iterations", type=int, default=8, help="復元時の反復回数")
    parser.add_argument("--scale", type=int, default=1, help="アップスケール倍率")
    args = parser.parse_args()

    if args.mode == "compress":
        # 入力音源を読み込み各チャンネルを圧縮
        rate, data = load_wav(args.input)
        left_params = compress(
            data[:, 0], block_size=args.block_size, search_step=args.search_step
        )
        right_params = compress(
            data[:, 1], block_size=args.block_size, search_step=args.search_step
        )
        # dictをそのまま渡すと型検査で怒られるため、object配列に変換して保存
        np.savez(
            args.params,
            left=np.array(left_params, dtype=object),
            right=np.array(right_params, dtype=object),
            rate=rate,
        )
    elif args.mode == "decompress":
        # 保存しておいたパラメータから音源を復元
        npz = np.load(args.params, allow_pickle=True)
        left_params = npz["left"].item()
        right_params = npz["right"].item()
        rate = int(npz["rate"])
        left = decompress(left_params, iterations=args.iterations, scale=args.scale)
        right = decompress(right_params, iterations=args.iterations, scale=args.scale)
        stereo = cp.stack([left, right], axis=-1)
        save_wav(args.output, rate * args.scale, stereo)
    else:
        # パラメータから復元し、元音源との誤差を計算
        rate, original = load_wav(args.input)
        npz = np.load(args.params, allow_pickle=True)
        left_params = npz["left"].item()
        right_params = npz["right"].item()
        rate_p = int(npz["rate"])
        left = decompress(left_params, iterations=args.iterations, scale=args.scale)
        right = decompress(right_params, iterations=args.iterations, scale=args.scale)
        stereo = cp.stack([left, right], axis=-1)
        save_wav(args.output, rate_p * args.scale, stereo)
        mse = calc_mse(original, stereo)
        print(f"平均二乗誤差: {mse}")
