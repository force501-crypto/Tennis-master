"""Script for processing the videos into frames and flow frames"""

import os
# 导入 absl 库来处理命令行参数（如果你运行报错，请 pip install absl-py）
from absl import app
#from models.vision.flownet.run import generate_flows
from utils.video import video_to_frames

# 1. 恢复成处理全部 5 个视频的配置
def vid2img(videos=('V006','V010'), videos_dir='data/videos', frames_dir='data/frames'):
    """
    videos: 包含你下载好的所有 5 个视频文件名
    videos_dir: 明确告诉代码去 data/videos 文件夹找
    frames_dir: 切好的图片会自动放到 data/frames 下，并按视频名字隔开
    """
    for video in videos:
        # 自动循环拼接路径，例如: data/videos/V006.mp4, data/videos/V007.mp4...
        video_path = os.path.join(videos_dir, video + '.mp4')

        if not os.path.exists(video_path):
            print(f"❌ 找不到视频！代码尝试访问的路径是: {os.path.abspath(video_path)}，已跳过此视频。")
            continue

        print(f"✅ 成功找到视频，开始切帧: {video_path}")
        video_to_frames(video_path=video_path,
                        frames_dir=frames_dir,
                        chunk_size=1000)

# 2. 修改 main 函数
def main(_argv):
    print("--- 开始处理数据 ---")

    # 确保 data 目录下有 frames 文件夹
    if not os.path.exists('data/frames'):
        os.makedirs('data/frames')
        print("已创建 data/frames 文件夹")

    print("Step 1: Video to Images (正在从全部视频提取图片...)")
    vid2img()

    # 如果你现在不需要生成光流(Flow)，这两行可以保持注释状态
    # print("Step 2: Images to Flow")
    # img2flw()

    print("--- 所有的视频全部处理完成！ ---")

if __name__ == '__main__':
    # 使用 absl 的方式启动
    app.run(main)