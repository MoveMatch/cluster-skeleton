import webdataset as wds
import numpy as np
import shutil

def extract_relevent_data(samples, handler=wds.handlers.reraise_exception):
    """
    Extracts the keys we want from the samples
    """
    for sample in samples:
        try:
            correct_landmarks = sample["correct_landmarks.npy"]
            incorrect_landmarks = sample["incorrect_landmarks.npy"]
            movement = sample["metadata.json"]["movement"]
            category = sample["metadata.json"]["category"]
            annotations = sample["metadata.json"]["annotations"]
            yield {
                "correct_landmarks": correct_landmarks,
                "incorrect_landmarks": incorrect_landmarks,
                "movement": movement,
                "category": category,
                "annotations": annotations,
                "metadata": sample["metadata.json"],
                "__key__": sample["__key__"],
            }
        except Exception as e:
            # This pattern here allows the handler to decide whether we can recover from the exception or not
            if handler(e):
                continue
            else:
                break
data_extracter = wds.filters.pipelinefilter(extract_relevent_data)


class MotionPerturbationDataset(wds.DataPipeline, wds.compat.FluidInterface):
    """
    A fluid interface wrapper that returns the correct and incorrect landmarks, the movement and category, and the annotations
    """

    def __init__(
        self,
        urls,
        handler=wds.handlers.reraise_exception,
        resample=False,
        shuffle_shards=False
    ):
        """
        Args:
            urls: List of strings (The urls to the webdataset)
            handler: Function (The handler to use when an exception is raised)
            resample: bool (Whether to resample the dataset). If True, each epoch is infinite unless explicitly stopped.
            shuffle_shards: bool (Whether to shuffle the shards)
        """
        super().__init__()
        if (isinstance(urls, str) and "s3:" in urls) or (isinstance(urls, list) and any(["s3:" in url for url in urls])):
            # Then this has an s3 link for the webdataset and we need extra packages
            if shutil.which("s3cmd") is None:
                raise RuntimeError("s3cmd is required for s3 webdataset")

        if resample:
            assert not shuffle_shards, "Cannot both resample and shuffle"
            self.append(wds.ResampledShards(urls))
        else:
            self.append(wds.SimpleShardList(urls))
            if shuffle_shards:
                self.append(wds.filters.shuffle(1000))

        self.append(wds.tarfile_to_samples(handler=handler))  # Extracts the tarfiles
        self.append(wds.decode())  # Turns the json into dicts and npy files into np.ndarrays
        self.append(data_extracter())  # Extracts the data we want

def get_mistake_horizons(samples, horizon, handler=wds.handlers.reraise_exception):
    """
    For frame in sample that has an annotation associated with it, we want to get the previous horizon frames and the list of feedback that could apply.
    If there are not enough frames before the frame, we will pad with the first frame.
    We expect the input to be the output of extract_relevent_data
    """
    for sample in samples:
        try:
            annotations = sample["annotations"]
            annotated_frames = {}
            for annotation in annotations:
                for frame in range(annotation["start_frame"], annotation["end_frame"]):
                    if frame not in annotated_frames:
                        annotated_frames[frame] = []
                    annotated_frames[frame].extend(annotation["sentences"])
            
            correct_landmarks = sample["correct_landmarks"]
            incorrect_landmarks = sample["incorrect_landmarks"]
            for i, (frame, sentences) in enumerate(annotated_frames.items()):
                if frame > horizon:
                    yield {
                        "correct_landmarks": correct_landmarks[frame-horizon:frame],
                        "incorrect_landmarks": incorrect_landmarks[frame-horizon:frame],
                        "sentences": sentences,
                        "frame": frame,
                        "category": sample["category"],
                        "__key__": sample["__key__"] + f"_{i}",
                    }
                else:
                    # Then we pad the start of the sequence with the first frame
                    extracted_correct_landmarks = correct_landmarks[0:frame]
                    extracted_incorrect_landmarks = incorrect_landmarks[0:frame]
                    padded_correct_landmarks = np.concatenate([correct_landmarks[0:1] for _ in range(horizon - frame)], axis=0)
                    padded_incorrect_landmarks = np.concatenate([incorrect_landmarks[0:1] for _ in range(horizon - frame)], axis=0)
                    yield {
                        "correct_landmarks": np.concatenate([padded_correct_landmarks, extracted_correct_landmarks], axis=0),
                        "incorrect_landmarks": np.concatenate([padded_incorrect_landmarks, extracted_incorrect_landmarks], axis=0),
                        "sentences": sentences,
                        "frame": frame,
                        "category": sample["category"],
                        "__key__": sample["__key__"] + f"_{i}",
                    }
        except Exception as e:
            # This pattern here allows the handler to decide whether we can recover from the exception or not
            if handler(e):
                continue
            else:
                break
make_mistake_horizons = wds.filters.pipelinefilter(get_mistake_horizons)

class MotionMistakeDataset(MotionPerturbationDataset):
    """
    Wraps the MotionPerturbationDataset, but returns
    """

    def __init__(
        self,
        urls,
        mistake_horizon=20, # The number of frames to look back when extracting the context for this mistake
        handler=wds.handlers.reraise_exception,
        resample=False,
        shuffle_shards=False,
        shuffle_samples=False,
    ):
        """
        Args:
            urls: List of strings (The urls to the webdataset)
            mistake_horizon: int (The number of frames to look back when extracting the context for a mistake)
            handler: Function (The handler to use when an exception is raised)
            resample: bool (Whether to resample the dataset). If True, each epoch is infinite unless explicitly stopped.
            shuffle_shards: bool (Whether to shuffle the shards)
            shuffle_samples: bool (Whether to shuffle the samples)
        """
        super().__init__(urls, handler=handler, resample=resample, shuffle_shards=shuffle_shards)
        self.append(make_mistake_horizons(mistake_horizon))
        if shuffle_samples:
            self.append(wds.filters.shuffle(1000))


if __name__ == "__main__":
    from pathlib import Path
    num_shards = len(list(Path("data/webdataset").glob("*.tar")))
    dataset_shards = "data/webdataset/{}.tar".format(f"{{0..{num_shards-1}}}")
    dataset = MotionPerturbationDataset(dataset_shards)
    for sample in dataset:
        print(sample["__key__"])

    dataset = MotionMistakeDataset(dataset_shards, mistake_horizon=40, shuffle_samples=True, resample=False)
    for sample in dataset:
        print(f"Key: {sample['__key__']}, Frame: {sample['frame']}, Landmarks Shape: {sample['correct_landmarks'].shape}, Sentences: {sample['sentences']}")

    from torch.utils.data import DataLoader
    dataset = MotionMistakeDataset(dataset_shards, mistake_horizon=40, shuffle_samples=True, resample=False)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=2)
    for batch in dataloader:
        print(f"After batching - Key: {batch['__key__']}, Frame: {batch['frame']}, Landmarks Shape: {batch['correct_landmarks'].shape}, Sentences: {batch['sentences']}")