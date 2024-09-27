import glob
import os
from typing import Generator
import ffmpeg
from PIL.Image import Image, frombuffer


class VideoDataset:
    def __init__(self, folder_path: str, fps: float=0.1,):
        self.folder_path = folder_path
        self.fps = fps
        self.video_files = glob.glob(os.path.join(folder_path, '**', '*.mp4'), recursive=True) + \
                           glob.glob(os.path.join(folder_path, '**', '*.mkv'), recursive=True)

    def _get_inp_stream(self, path): return (ffmpeg.input(path,).filter('fps', fps=self.fps))

    def _read_frames_one_by_one(self, video_path):
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
            width = int(video_info['width'])
            height = int(video_info['height'])
            
            process = (self._get_inp_stream(video_path).output('pipe:', format='rawvideo', pix_fmt='rgb24').run_async(pipe_stdout=True, pipe_stderr=True))

            while True:
                in_bytes = process.stdout.read(width * height * 3)
                if not in_bytes: break
                yield frombuffer("RGB", (width, height), in_bytes,)
            process.wait()

        except ffmpeg.Error as e:
            print(f"Error reading video {video_path}: {e}")
            return []

    def __iter__(self) -> Generator[Image]:
        for video_path in self.video_files:
            for frame in self._read_frames_one_by_one(video_path): yield frame
