"""
clipper.py
视频片段裁剪与合并模块
使用 ffmpeg-python，速度快，不重新编码
"""
import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _ffmpeg_cmd(*args) -> list[str]:
    return ["ffmpeg", "-y", *args]


def clip_segment(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    progress_callback=None,
) -> bool:
    """
    从 video_path 截取 [start_sec, end_sec] 片段，输出到 output_path
    使用 stream copy 不重新编码，速度极快
    返回是否成功
    """
    duration = max(0.1, end_sec - start_sec)
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = _ffmpeg_cmd(
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        output_path,
    )
    logger.debug(f"裁剪命令: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error(f"ffmpeg 错误: {result.stderr.decode(errors='ignore')}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg 超时")
        return False
    except FileNotFoundError:
        logger.error("找不到 ffmpeg，请先安装 ffmpeg")
        raise


def concat_clips(
    clip_paths: list[str],
    output_path: str,
    progress_callback=None,
) -> bool:
    """
    将多个片段合并为一个视频
    使用 ffmpeg concat demuxer，不重新编码
    返回是否成功
    """
    if not clip_paths:
        logger.warning("没有片段可合并")
        return False

    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # 写临时 concat 文件
    concat_list_path = output_path + ".concat_list.txt"
    try:
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for p in clip_paths:
                # ffmpeg concat 要求绝对路径或相对路径，特殊字符需转义
                safe_p = str(Path(p).resolve()).replace("'", "'\\''")
                f.write(f"file '{safe_p}'\n")

        cmd = _ffmpeg_cmd(
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            output_path,
        )
        logger.debug(f"合并命令: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,
        )
        if result.returncode != 0:
            logger.error(f"ffmpeg 合并错误: {result.stderr.decode(errors='ignore')}")
            return False
        return True
    except Exception as e:
        logger.error(f"合并失败: {e}")
        return False
    finally:
        if os.path.exists(concat_list_path):
            os.remove(concat_list_path)


def get_video_info(video_path: str) -> dict:
    """
    用 ffprobe 获取视频基本信息
    返回 {duration, fps, width, height}
    """
    import json
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            return {}
        info = json.loads(result.stdout)
        video_stream = next(
            (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
            None
        )
        if not video_stream:
            return {}
        fps_str = video_stream.get("r_frame_rate", "30/1")
        try:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        except Exception:
            fps = 30.0
        return {
            "duration": float(info.get("format", {}).get("duration", 0)),
            "fps": fps,
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
        }
    except Exception as e:
        logger.warning(f"ffprobe 失败: {e}")
        return {}


def extract_thumbnail(video_path: str, timestamp: float, output_path: str) -> bool:
    """从视频指定时间点提取缩略图（JPEG）"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = _ffmpeg_cmd(
        "-ss", str(timestamp),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        output_path,
    )
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        return result.returncode == 0
    except Exception as e:
        logger.warning(f"缩略图提取失败: {e}")
        return False
