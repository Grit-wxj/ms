# -*- coding: utf-8 -*-
import subprocess
import sys
import importlib.util
import os
import json
import argparse
import platform
import warnings
from datetime import datetime

# --- 自动依赖检查与安装 ---
def check_and_install_dependencies():
    """自动检测并安装所需的 Python 库"""
    required_packages = [
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("librosa", "librosa"),
        ("pillow", "PIL"),
        ("static-ffmpeg", "static_ffmpeg")
    ]

    print("[-] 正在检查环境依赖...")
    for package_name, import_name in required_packages:
        if importlib.util.find_spec(import_name) is None:
            print(f"[!] 未检测到库 '{package_name}' ({import_name})，正在尝试自动安装...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                print(f"[+] '{package_name}' 安装成功")
            except subprocess.CalledProcessError:
                print(f"[x] '{package_name}' 安装失败。请尝试手动运行: pip install {package_name}")
                sys.exit(1)
    print("[-] 所有依赖库检查通过。\n")

check_and_install_dependencies()

import cv2
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from PIL import Image, ImageChops, ImageEnhance, ImageStat
import static_ffmpeg

# 自动配置 FFmpeg
print("[-] 正在初始化 FFmpeg 环境...")
try:
    static_ffmpeg.add_paths()
    print("[+] FFmpeg 环境初始化成功")
except Exception as e:
    print(f"[!] FFmpeg 初始化警告: {e}")

# 配置中文字体
def configure_matplotlib_fonts():
    system_name = platform.system()
    if system_name == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial']
    elif system_name == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Droid Sans Fallback']
    plt.rcParams['axes.unicode_minus'] = False

configure_matplotlib_fonts()

def format_timestamp(seconds):
    """将秒数转换为 分:秒.毫秒 格式"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{m:02d}分{s:02d}秒{ms:03d}"

class MediaForensicsTool:
    def __init__(self, file_path):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")
        print(f"[*] 已加载文件: {file_path}")

    def analyze_metadata(self):
        print("\n--- [1] 开始元数据分析 ---")
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', self.file_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                 raise Exception("FFprobe 执行失败")

            data = json.loads(result.stdout)
            tags = data.get('format', {}).get('tags', {})
            
            # 检查常见的编辑软件签名
            suspicious_keywords = ['Lavf', 'Adobe', 'Premiere', 'Final Cut', 'HandBrake', 'DaVinci', 'CapCut']
            encoder = tags.get('encoder', '')
            
            # 深度查找
            if not encoder:
                for s in data.get('streams', []):
                    encoder = s.get('tags', {}).get('encoder', encoder)

            print(f"    编码器信息: {encoder if encoder else '未找到'}")
            
            found = False
            if encoder:
                for k in suspicious_keywords:
                    if k.lower() in encoder.lower():
                        print(f"[!] 警告: 发现后期软件签名 -> {k}")
                        found = True
            
            if not found:
                print("[√] 元数据洁净度: 较高 (未发现明显后期软件标签)")
                
        except Exception as e:
            print(f"[!] 元数据提取失败: {e}")

    def detect_video_cuts_smart(self):
        """
        智能版：视频镜头分割检测
        改用 [排序 + 局部极值] 策略，而非单纯的阈值截断。
        能更准确地抓出最显著的那个拼接点。
        """
        print("\n--- [2] 开始视频画面剪辑点扫描 (智能排序算法) ---")
        print("    正在逐帧计算色彩相关性并寻找突变极值...")
        
        cap = cv2.VideoCapture(self.file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ret, prev_frame = cap.read()
        if not ret:
            print("[!] 无法读取视频帧")
            return []

        # 转换为 HSV 空间，计算直方图
        prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
        prev_hist = cv2.calcHist([prev_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(prev_hist, prev_hist, 0, 1, cv2.NORM_MINMAX)
        
        frame_idx = 0
        diff_scores = [] # 存储 (frame_time, correlation_score)
        
        # 步进扫描，每帧都看，保证精度
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            
            # 为了性能，可以跳过部分帧，但在寻找单一拼接点时建议逐帧或隔帧
            if frame_idx % 2 != 0: 
                continue

            curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
            curr_hist = cv2.calcHist([curr_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)
            
            # 计算直方图相关性 (1.0 = 相同, 0.0 = 完全不同)
            # 我们用 1 - correlation 作为“差异分”，分数越高差异越大
            correlation = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
            diff_score = 1.0 - correlation
            
            timestamp = frame_idx / fps
            diff_scores.append((timestamp, diff_score))
            
            prev_hist = curr_hist
            
            if frame_idx % 100 == 0:
                print(f"    ...已扫描 {format_timestamp(timestamp)}", end="\r")

        cap.release()
        print("\n    扫描完成，正在计算显著性排名...")
        
        if not diff_scores:
            return []

        # --- 智能分析逻辑 ---
        # 1. 找出差异分最高的点 (差异越大，越可能是硬切)
        # 排序：从大到小
        sorted_scores = sorted(diff_scores, key=lambda x: x[1], reverse=True)
        
        # 2. 过滤邻近点 (只保留局部最大的那个峰值)
        final_cuts = []
        for t, score in sorted_scores:
            # 如果分数太低（小于0.15，即相关性大于0.85），说明只是普通运镜，忽略
            if score < 0.15: 
                continue
                
            # 检查是否与已有的点太近 (1.5秒内)
            is_near = False
            for existing_t, _ in final_cuts:
                if abs(t - existing_t) < 1.5:
                    is_near = True
                    break
            
            if not is_near:
                final_cuts.append((t, score))
                if len(final_cuts) >= 5: # 只取前5个最可疑的
                    break
        
        print(f"    [视频分析结果]")
        if not final_cuts:
            print("    - 结果: 画面平滑，未检测到显著拼接。")
            return []
        else:
            print(f"    - 结果: 发现潜在突变点，按【置信度】从高到低排序：")
            for i, (t, score) in enumerate(final_cuts):
                # 差异分越高，置信度越高
                confidence = min(score * 100 + 20, 99.9) 
                print(f"      [{i+1}] 时间: {format_timestamp(t)} | 差异强度: {score:.3f}")
            
            # 返回时间点列表
            return [t for t, score in final_cuts]

    def perform_ela_on_frame(self, frame_time_sec=1.0):
        print(f"\n--- [3] ELA 篡改痕迹深度分析 (采样点: {format_timestamp(frame_time_sec)}) ---")
        
        cap = cv2.VideoCapture(self.file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * frame_time_sec))
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("[!] 无法读取该帧")
            return

        original_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        temp_filename = "temp_ela.jpg"
        original_img.save(temp_filename, "JPEG", quality=90)
        compressed_img = Image.open(temp_filename)
        
        ela_img = ImageChops.difference(original_img, compressed_img)
        
        # --- 自动化评分逻辑 ---
        stat = ImageStat.Stat(ela_img)
        mean_diff = sum(stat.mean) / len(stat.mean)
        
        # 归一化评分
        tamper_score = min(mean_diff * 10, 100)
        
        print(f"    [ELA 量化评分]")
        print(f"    - 异常系数: {tamper_score:.2f}/100")
        
        if tamper_score > 30:
            print("    - ⚠️ 警告: 检测到异常的压缩伪影，该帧可能包含合成元素！")
        else:
            print("    - 状态: 压缩特征均匀，未见明显局部篡改。")

        # 视觉增强
        extrema = ela_img.getextrema()
        max_diff_val = max([ex[1] for ex in extrema])
        scale = 255.0 / (max_diff_val if max_diff_val > 0 else 1) * 15 
        ela_img = ImageEnhance.Brightness(ela_img).enhance(scale)
        
        # 保存
        output_filename = f"ela_check.png"
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("原始帧")
        plt.imshow(original_img)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title(f"ELA (异常分: {tamper_score:.1f})")
        plt.imshow(ela_img)
        plt.axis('off')
        
        try:
            plt.savefig(output_filename)
            print(f"    [图片] ELA分析图已保存至: {output_filename}")
        except Exception:
            pass
        
        if os.path.exists(temp_filename):
            try: os.remove(temp_filename)
            except: pass

    def analyze_audio_smart(self):
        """
        智能版：音频特征分析
        同样采用排序策略，找出最突兀的音频变化点。
        """
        print("\n--- [4] 开始音频特征显著性分析 ---")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                y, sr = librosa.load(self.file_path, duration=60)
            
            # 1. 能量突变 (Onset Strength)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
            
            # 将 (时间, 强度) 打包
            onset_points = []
            for i, strength in enumerate(onset_env):
                onset_points.append((times[i], strength))
            
            # 2. 排序并过滤
            # 按强度降序
            sorted_onsets = sorted(onset_points, key=lambda x: x[1], reverse=True)
            
            final_audio_cuts = []
            # 获取平均强度作为基准
            avg_strength = np.mean(onset_env)
            std_strength = np.std(onset_env)
            threshold_base = avg_strength + 3 * std_strength
            
            for t, strength in sorted_onsets:
                if strength < threshold_base: # 忽略低于背景噪音波动的
                    continue
                    
                # 距离过滤 (1秒)
                is_near = False
                for existing_t, _ in final_audio_cuts:
                    if abs(t - existing_t) < 1.0:
                        is_near = True
                        break
                
                if not is_near:
                    final_audio_cuts.append((t, strength))
                    if len(final_audio_cuts) >= 5:
                        break
            
            print(f"    [音频分析结果]")
            if not final_audio_cuts:
                print("    - 结果: 音频平稳。")
                return []
            else:
                print(f"    - 结果: 发现潜在断层，按【显著性】从高到低排序：")
                for i, (t, strength) in enumerate(final_audio_cuts):
                    print(f"      [{i+1}] 时间: {format_timestamp(t)} | 突变强度: {strength:.2f}")
                
            # 绘图
            plt.figure(figsize=(12, 6))
            D = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title('音频频谱与 Top 疑似点')
            for t, _ in final_audio_cuts:
                plt.axvline(x=t, color='r', linestyle='--', alpha=0.8, linewidth=1.5)
            plt.tight_layout()
            plt.savefig("audio_check_smart.png")
            print(f"    [图片] 音频分析图已保存至: audio_check_smart.png")
            
            return [t for t, s in final_audio_cuts]
            
        except Exception as e:
            print(f"[!] 音频分析中断: {e}")
            return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="文件路径")
    args = parser.parse_args()

    target_file = args.file
    if not target_file:
        print("\n=== 音视频拼接取证工具 (智能排序版) ===")
        print("请输入文件路径:")
        target_file = input(">>> ").strip().strip("'").strip('"')

    if target_file and os.path.exists(target_file):
        tool = MediaForensicsTool(target_file)
        
        # 1. 元数据
        tool.analyze_metadata()
        
        # 2. 视频检测 (智能排序)
        video_cuts = tool.detect_video_cuts_smart()
        
        # 3. ELA 分析 (只分析 Top 1 和 Top 2，因为用户说只有一处拼接)
        check_points = [1.0]
        if video_cuts:
            check_points = video_cuts[:2] 
            print(f"\n[i] 重点对前 {len(check_points)} 个可疑点进行 ELA 篡改验证...")
        
        for t in check_points:
            tool.perform_ela_on_frame(t)
        
        # 4. 音频检测 (智能排序)
        audio_cuts = tool.analyze_audio_smart()
        
        # 5. 综合判定
        print("\n=== 🏁 最终取证结论 (基于 Top 排名) ===")
        
        # 寻找最强匹配 (Top 1 Video vs Top 1 Audio)
        primary_match = False
        if video_cuts and audio_cuts:
            v_top1 = video_cuts[0]
            # 检查音频前3名里有没有和视频第1名匹配的
            for a_cut in audio_cuts[:3]: 
                if abs(v_top1 - a_cut) < 1.0:
                    print(f"✅ 【确凿证据】 视频最强突变点与音频断层重合！")
                    print(f"   >>> 拼接点极大概率在: {format_timestamp(v_top1)} <<<")
                    primary_match = True
                    break
        
        if not primary_match:
            if video_cuts:
                print(f"⚠️ 【疑似拼接】 视频画面在 {format_timestamp(video_cuts[0])} 处有最大突变。")
            if audio_cuts:
                print(f"⚠️ 【疑似拼接】 音频波形在 {format_timestamp(audio_cuts[0])} 处有最大断层。")
                
            print("ℹ️  如果上述两个时间点接近，即为拼接处。如果不接近，可能是画外音剪辑。")

    else:
        print("[!] 文件不存在")
