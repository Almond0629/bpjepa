import gc
import io
import shutil
import zipfile
from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from torch import Tensor
from torch.utils.data import Dataset


def load_features(
    features_dir: str | Path,
    video_paths: List[str | Path],
    filename: str,
    video_zip_prefix: str = "",
    as_tensor: bool = False,
) -> dict[int, dict[int, np.ndarray | Tensor]]:
    """Load SoccerNet features.

    Args:
        features_dir: Directory or zip where the features are stored.
        video_paths: Path to the videos in directory.
        filename: Name of the files containing the features.
        video_zip_prefix: Video zip prefix if features stored in Zip.
        as_tensor: Whether to decode the features as tensor or numpy.

    Returns:
        The loaded features for each video and half.
    """
    features_dir = Path(features_dir)
    video_zip_prefix = Path(video_zip_prefix)

    video_features = {video_index: {} for video_index in range(len(video_paths))}
    from_zip = zipfile.is_zipfile(features_dir)

    for video_index, video_path in enumerate(video_paths):
        if from_zip:
            with zipfile.ZipFile(features_dir, "r") as z:
                for half_index in range(1, 3):
                    with z.open(
                        str(
                            video_zip_prefix
                            / video_path
                            / f"{half_index}_{filename}.npy"
                        )
                    ) as f:
                        video_features[video_index][half_index] = np.load(f)
                        if as_tensor:
                            video_features[video_index][half_index] = torch.from_numpy(
                                video_features[video_index][half_index]
                            )
        else:
            for half_index in range(1, 3):
                video_features[video_index][half_index] = np.load(
                    features_dir / video_path / f"{half_index}_{filename}.npy"
                )
                if as_tensor:
                    video_features[video_index][half_index] = torch.from_numpy(
                        video_features[video_index][half_index]
                    )

    return video_features


def save_features(
    dataset: Dataset,
    saving_path: str | Path,
    features: dict[int, dict[int, Tensor]],
    filename: str,
    make_zip: bool,
) -> None:
    """Save the features, one file per half per match.

    Args:
        dataset: Dataset to save the features from.
        saving_path: Path to save the features.
        features: The features to save.
        filename: The filename for each file containing the stored features.
        make_zip: Store the features as a Zip file.
    """
    saving_path = Path(saving_path)

    for video_index in features:
        video_path = saving_path / str(
            dataset.get_video_metadata(video_index)["url_local"]
        )
        video_path.mkdir(exist_ok=True, parents=True)
        for half_index in features[video_index]:
            half_path = f"{half_index}_{filename}.npy"
            np.save(video_path / half_path, features[video_index][half_index])

    if make_zip:
        shutil.make_archive(str(saving_path), "zip", saving_path)
        shutil.rmtree(saving_path)

    return


def pca_features(
    features: dict[int, dict[int, np.ndarray]],
    dim: int,
    standardize: bool = True,
    **kwargs,
):
    """Apply PCA on the given SoccerNet features.

    Args:
        features: The features to apply PCA on.
        dim: The output dimension of the PCA.
        standardize: Whether to standardize features before PCA.

    Returns:
        The dimensionally reduced features. The input mapping is consumed to
        keep peak memory bounded while its arrays are copied into one buffer.
    """
    layout = []
    num_rows = 0
    num_features = None
    dtype = None

    for video_index, halves in features.items():
        for half_index, feature in halves.items():
            if feature.ndim != 2:
                raise ValueError(
                    f"Expected 2D features, got shape {feature.shape} for "
                    f"video {video_index}, half {half_index}."
                )
            if num_features is None:
                num_features = feature.shape[1]
            elif feature.shape[1] != num_features:
                raise ValueError(
                    f"Inconsistent feature dimension {feature.shape[1]} for "
                    f"video {video_index}, half {half_index}; expected "
                    f"{num_features}."
                )
            layout.append((video_index, half_index, num_rows, feature.shape[0]))
            num_rows += feature.shape[0]
            dtype = (
                feature.dtype
                if dtype is None
                else np.result_type(dtype, feature.dtype)
            )

    if not layout:
        raise ValueError("Cannot apply PCA to an empty feature mapping.")

    features_all = np.empty((num_rows, num_features), dtype=dtype)
    feature_sum = np.zeros(num_features, dtype=np.float64)
    feature_square_sum = np.zeros(num_features, dtype=np.float64)

    for video_index, half_index, start, length in layout:
        feature = features[video_index].pop(half_index)
        feature_slice = features_all[start : start + length]
        feature_slice[...] = feature
        if standardize:
            feature_sum += feature_slice.sum(axis=0, dtype=np.float64)
            feature_square_sum += np.einsum(
                "ij,ij->j", feature_slice, feature_slice, dtype=np.float64
            )
        del feature

    features.clear()
    gc.collect()

    if standardize:
        mean = feature_sum / num_rows
        variance = feature_square_sum / num_rows - np.square(mean)
        scale = np.sqrt(np.maximum(variance, 0))
        scale[scale == 0] = 1
        features_all -= mean
        features_all /= scale

    kwargs.setdefault("copy", False)
    kwargs.setdefault("svd_solver", "randomized")
    pca = PCA(n_components=dim, **kwargs)
    features_all = pca.fit_transform(features_all)

    features_pca = {video_index: {} for video_index, _, _, _ in layout}
    for video_index, half_index, start, length in layout:
        features_pca[video_index][half_index] = features_all[start : start + length]

    return features_pca


def sampled_pca_features_to_zip(
    features_dir: str | Path,
    video_paths: List[str | Path],
    filename: str,
    saving_path: str | Path,
    dim: int,
    sample_per_half: int = 256,
    transform_batch_size: int = 512,
    seed: int = 42,
    video_zip_prefix: str = "",
) -> Path:
    """Fit sampled PCA and transform SoccerNet features without loading them all.

    Normalization statistics are computed from every feature row. PCA is fitted
    on a uniform sample from every half, then all rows are transformed in
    batches and written directly to a ZIP archive.
    """
    features_dir = Path(features_dir)
    video_zip_prefix = Path(video_zip_prefix)
    feature_refs = [
        (Path(video_path), half_index)
        for video_path in video_paths
        for half_index in range(1, 3)
    ]
    rng = np.random.default_rng(seed)
    input_archive = (
        zipfile.ZipFile(features_dir, "r")
        if zipfile.is_zipfile(features_dir)
        else None
    )

    def load_feature(video_path: Path, half_index: int) -> np.ndarray:
        feature_path = video_path / f"{half_index}_{filename}.npy"
        if input_archive is not None:
            with input_archive.open(str(video_zip_prefix / feature_path)) as file:
                return np.load(file)
        return np.load(features_dir / feature_path, mmap_mode="r")

    try:
        metadata = []
        num_features = None
        num_samples = 0
        for video_path, half_index in feature_refs:
            feature = load_feature(video_path, half_index)
            if feature.ndim != 2:
                raise ValueError(
                    f"Expected 2D features, got shape {feature.shape} for "
                    f"{video_path}, half {half_index}."
                )
            if num_features is None:
                num_features = feature.shape[1]
            elif feature.shape[1] != num_features:
                raise ValueError(
                    f"Inconsistent feature dimension {feature.shape[1]} for "
                    f"{video_path}, half {half_index}; expected {num_features}."
                )
            half_samples = min(sample_per_half, feature.shape[0])
            metadata.append(
                (video_path, half_index, feature.shape[0], half_samples)
            )
            num_samples += half_samples
            del feature

        samples = np.empty((num_samples, num_features), dtype=np.float32)
        feature_sum = np.zeros(num_features, dtype=np.float64)
        feature_square_sum = np.zeros(num_features, dtype=np.float64)
        total_rows = 0
        sample_start = 0

        print(f"statistics_and_sampling files={len(metadata)} samples={num_samples}")
        for file_index, (video_path, half_index, rows, half_samples) in enumerate(
            metadata, start=1
        ):
            feature = load_feature(video_path, half_index)
            indices = rng.choice(rows, size=half_samples, replace=False)
            samples[sample_start : sample_start + half_samples] = feature[indices]
            sample_start += half_samples

            for start in range(0, rows, transform_batch_size):
                feature_batch = feature[start : start + transform_batch_size]
                feature_sum += feature_batch.sum(axis=0, dtype=np.float64)
                feature_square_sum += np.einsum(
                    "ij,ij->j", feature_batch, feature_batch, dtype=np.float64
                )
            total_rows += rows
            del feature
            if file_index % 50 == 0 or file_index == len(metadata):
                print(f"statistics_and_sampling {file_index}/{len(metadata)}")

        mean = (feature_sum / total_rows).astype(np.float32)
        variance = feature_square_sum / total_rows - np.square(
            mean.astype(np.float64)
        )
        scale = np.sqrt(np.maximum(variance, 0)).astype(np.float32)
        scale[scale == 0] = 1
        samples -= mean
        samples /= scale

        print(f"fit_pca rows={num_samples} input_dim={num_features} dim={dim}")
        pca = PCA(
            n_components=dim,
            copy=False,
            svd_solver="randomized",
            random_state=seed,
        )
        pca.fit(samples)
        del samples
        gc.collect()

        output_filename = f"pca_{dim}_std_{filename}"
        output_zip = Path(saving_path) / f"{output_filename}.zip"
        output_zip.parent.mkdir(exist_ok=True, parents=True)
        if output_zip.exists():
            raise FileExistsError(f"Refusing to overwrite {output_zip}.")

        print(f"transform_and_save output={output_zip}")
        with zipfile.ZipFile(
            output_zip, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as output_archive:
            for file_index, (video_path, half_index, rows, _) in enumerate(
                metadata, start=1
            ):
                feature = load_feature(video_path, half_index)
                transformed = np.empty((rows, dim), dtype=np.float32)
                for start in range(0, rows, transform_batch_size):
                    feature_batch = np.array(
                        feature[start : start + transform_batch_size],
                        dtype=np.float32,
                        copy=True,
                    )
                    feature_batch -= mean
                    feature_batch /= scale
                    transformed[start : start + len(feature_batch)] = pca.transform(
                        feature_batch
                    )
                buffer = io.BytesIO()
                np.save(buffer, transformed, allow_pickle=False)
                archive_path = (
                    video_path / f"{half_index}_{output_filename}.npy"
                ).as_posix()
                output_archive.writestr(archive_path, buffer.getbuffer())
                del feature, transformed, buffer
                if file_index % 50 == 0 or file_index == len(metadata):
                    print(f"transform_and_save {file_index}/{len(metadata)}")

        return output_zip
    finally:
        if input_archive is not None:
            input_archive.close()
