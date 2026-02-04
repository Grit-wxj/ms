音视频取证工具箱 (Media Forensics Tool)

这是一个基于 Python 的音视频电子数据取证工具，专为快速检测视频拼接、音频断层、画面篡改（Deepfake/P图）及元数据异常而设计。

它集成了计算机视觉（OpenCV）、数字信号处理（Librosa/MFCC）和图像取证（ELA）技术，采用“显著性排序”算法，能够自动定位最可疑的伪造点，而非依赖死板的阈值。

核心功能

视频镜头分割检测 (Video Cut Detection)

使用 HSV 色彩直方图相关性算法，抗光照干扰。

自动识别画面硬切、转场，并按置信度排序输出。

音频声纹断层分析 (Audio Forensics)

双重检测：结合能量突变 (Onset Strength) 和 MFCC 声纹特征 (Timbre/Environment)。

能识别出“音量未变但录音环境改变”的精细拼接。

ELA 篡改痕迹分析 (Error Level Analysis)

对可疑帧进行错误级别分析，检测是否存在后期合成（P图）痕迹。

提供量化的“篡改风险评分”。

元数据指纹筛查 (Metadata Check)

自动提取并分析编码器标签。

识别常见后期软件（Premiere, CapCut, Lavf, LAME等）的残留痕迹。

环境依赖

工具内置了依赖检查和自动配置功能。

Python 版本: 3.8+

必要库:

opencv-python (视频处理)

numpy (数值计算)

matplotlib (绘图)

librosa (音频分析)

pillow (图像处理)

static-ffmpeg (自动管理 FFmpeg 二进制环境，无需手动配置系统 PATH)

首次运行时，脚本会自动尝试安装缺失的库。

使用方法

方式一：命令行模式

适用于批处理或明确知道文件路径的情况。

python media_forensics.py -f "path/to/video.mp4"


方式二：交互模式

直接运行脚本，根据提示拖入文件。

python media_forensics.py


程序启动后会提示： 请输入文件路径 (支持 MP4, AVI, MP3, WAV 等):

支持格式

视频: MP4, AVI, MOV, MKV 等常见格式。

音频: MP3, WAV, M4A, AAC, FLAC 等。

注：对于纯音频文件，工具会自动跳过画面分析模块。

输出解读

程序运行结束后，会在控制台输出分析报告，并在当前目录生成可视化图表：

控制台报告:

[元数据]: 显示是否发现剪辑软件签名。

[视频分析]: 列出 Top 5 可疑画面突变时间点。

[音频分析]: 列出 Top 8 可疑声纹断层时间点。

[结论]: 综合判读风险等级（高/中/低）。

可视化图片:

ela_check.png: 可疑帧的 ELA 热力图（高亮区域即为疑似篡改）。

audio_check_smart.png: 音频波形与声纹突变曲线图（绿色峰值即为断层）。

免责声明

本工具仅供技术研究与辅助参考，自动化分析结果不能作为唯一的法律证据。在进行司法取证时，请结合人工审查及专业商业软件（如 Amped FIVE）进行复核。
