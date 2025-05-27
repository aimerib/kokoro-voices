# modules/downloader.py
"""YouTube audio downloader module"""

from pathlib import Path
from typing import Optional, List

from utilities import get_module_logger

import yt_dlp


class AudioDownloader:
    """Download audio from YouTube videos"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_module_logger(__name__)

        # yt-dlp options
        self.ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": str(self.output_dir / "%(title)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
        }

    def download(self, url: str) -> Optional[Path]:
        """Download audio from YouTube URL"""
        try:
            self.logger.info("Downloading: %s", url)

            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                # Replace extension with mp3
                audio_path = Path(filename).with_suffix(".mp3")

                if audio_path.exists():
                    self.logger.info("Downloaded: %s", audio_path.name)
                    return audio_path
                else:
                    self.logger.error("Download failed: %s", url)
                    return None

        except Exception as e:  # pylint: disable=broad-except
            self.logger.error("Error downloading %s: %s", url, e)
            return None

    def download_playlist(self, playlist_url: str) -> List[Path]:
        """Download all videos from a playlist"""
        downloaded = []

        try:
            with yt_dlp.YoutubeDL({**self.ydl_opts, "extract_flat": True}) as ydl:
                playlist_info = ydl.extract_info(playlist_url, download=False)

                if "entries" in playlist_info:
                    self.logger.info(
                        "Found %s videos in playlist", len(playlist_info['entries'])
                    )

                    for entry in playlist_info["entries"]:
                        if "url" in entry:
                            result = self.download(entry["url"])
                            if result:
                                downloaded.append(result)

        except Exception as e:  # pylint: disable=broad-except
            self.logger.error("Error processing playlist: %s", e)

        return downloaded
