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
        
        # 媒体类型标志位
        self.has_video = False
        self.has_audio = False
        self.duration = 0

    def analyze_metadata(self):
        print("\n--- [1] 开始元数据及媒体类型分析 ---")
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', self.file_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                 raise Exception("FFprobe 执行失败")

            data = json.loads(result.stdout)
            format_info = data.get('format', {})
            tags = format_info.get('tags', {})
            self.duration = float(format_info.get('duration', 0))
            
            # 检测流类型
            streams = data.get('streams', [])
            for s in streams:
                if s['codec_type'] == 'video':
                    self.has_video = True
                elif s['codec_type'] == 'audio':
                    self.has_audio = True

            # 打印基础信息
            file_type = "未知"
            if self.has_video and self.has_audio: file_type = "视频 (含音频)"
            elif self.has_video: file_type = "纯视频 (无音频)"
            elif self.has_audio: file_type = "纯音频 (MP3/WAV等)"
            
            print(f"    检测类型: 【{file_type}】")
            print(f"    时长: {self.duration} 秒")
            print(f"    容器格式: {format_info.get('format_name')}")

            # 检查常见的编辑软件签名
            suspicious_keywords = ['Lavf', 'Adobe', 'Premiere', 'Final Cut', 'HandBrake', 'DaVinci', 'CapCut', 'LAME']
            encoder = tags.get('encoder', '')
            
            if not encoder:
                for s in streams:
                    encoder = s.get('tags', {}).get('encoder', encoder)

            print(f"    编码器信息: {encoder if encoder else '未找到'}")
            
            found = False
            if encoder:
                for k in suspicious_keywords:
                    if k.lower() in encoder.lower():
                        print(f"[!] 警告: 发现后期软件签名 -> {k}")
                        found = True
            
            if not found:
                print("[√] 元数据洁净度: 较高")
                
        except Exception as e:
            print(f"[!] 元数据提取失败: {e}")

    def detect_video_cuts_smart(self):
        """
        智能版：视频镜头分割检测
        """
        if not self.has_video:
            return []

        print("\n--- [2] 开始视频画面剪辑点扫描 (智能排序算法) ---")
        print("    正在逐帧计算色彩相关性并寻找突变极值...")
        
        cap = cv2.VideoCapture(self.file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        ret, prev_frame = cap.read()
        if not ret:
            print("[!] 无法读取视频帧")
            return []

        prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
        prev_hist = cv2.calcHist([prev_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(prev_hist, prev_hist, 0, 1, cv2.NORM_MINMAX)
        
        frame_idx = 0
        diff_scores = [] 
        
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            
            if frame_idx % 2 != 0: 
                continue

            curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
            curr_hist = cv2.calcHist([curr_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)
            
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

        sorted_scores = sorted(diff_scores, key=lambda x: x[1], reverse=True)
        
        final_cuts = []
        for t, score in sorted_scores:
            if score < 0.15: 
                continue
            is_near = False
            for existing_t, _ in final_cuts:
                if abs(t - existing_t) < 1.5:
                    is_near = True
                    break
            if not is_near:
                final_cuts.append((t, score))
                if len(final_cuts) >= 5: 
                    break
        
        print(f"    [视频分析结果]")
        if not final_cuts:
            print("    - 结果: 画面平滑，未检测到显著拼接。")
            return []
        else:
            print(f"    - 结果: 发现潜在突变点，按【置信度】从高到低排序：")
            for i, (t, score) in enumerate(final_cuts):
                confidence = min(score * 100 + 20, 99.9) 
                print(f"      [{i+1}] 时间: {format_timestamp(t)} | 差异强度: {score:.3f}")
            return [t for t, score in final_cuts]

    def perform_ela_on_frame(self, frame_time_sec=1.0):
        if not self.has_video:
            return

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
        stat = ImageStat.Stat(ela_img)
        mean_diff = sum(stat.mean) / len(stat.mean)
        tamper_score = min(mean_diff * 10, 100)
        
        print(f"    [ELA 量化评分]")
        print(f"    - 异常系数: {tamper_score:.2f}/100")
        
        if tamper_score > 30:
            print("    - ⚠️ 警告: 检测到异常的压缩伪影，该帧可能包含合成元素！")
        else:
            print("    - 状态: 压缩特征均匀，未见明显局部篡改。")

        extrema = ela_img.getextrema()
        max_diff_val = max([ex[1] for ex in extrema])
        scale = 255.0 / (max_diff_val if max_diff_val > 0 else 1) * 15 
        ela_img = ImageEnhance.Brightness(ela_img).enhance(scale)
        
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
        智能版：音频特征分析 (引入 MFCC 声学特征)
        增强对“音色/环境音”突变的检测能力。
        """
        if not self.has_audio:
            print("\n--- [4] 音频分析跳过 (无音频流) ---")
            return []

        print("\n--- [4] 开始音频特征显著性分析 (能量 + MFCC声纹) ---")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                # 加载音频
                duration_to_load = min(self.duration, 180) 
                y, sr = librosa.load(self.file_path, duration=duration_to_load)
            
            # --- 特征提取 ---
            
            # 1. Onset Strength (能量突变) - 捕捉硬剪辑
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            
            # 2. MFCC Delta (声纹/音色突变) - 捕捉环境变化
            # MFCC 反映了音频的音色特征，不同录音环境 MFCC 会有显著差异
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            # 计算每一帧与前一帧的差异 (Delta)
            mfcc_delta = librosa.feature.delta(mfcc)
            # 计算每帧变化的 L2 范数，得到一个标量序列
            mfcc_change = np.linalg.norm(mfcc_delta, axis=0)
            
            # --- 归一化与融合 ---
            
            # 调整长度一致 (MFCC 帧数通常比 Onset 少一点点，对齐一下)
            min_len = min(len(onset_env), len(mfcc_change))
            onset_env = onset_env[:min_len]
            mfcc_change = mfcc_change[:min_len]
            times = librosa.frames_to_time(np.arange(min_len), sr=sr)
            
            # 归一化到 0-1
            def normalize(arr):
                return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-6)
            
            norm_onset = normalize(onset_env)
            norm_mfcc = normalize(mfcc_change)
            
            # 综合评分：40% 能量权重 + 60% 音色权重
            # 加大 MFCC 权重有助于发现那些音量没变但音色变了的拼接
            combined_score = 0.4 * norm_onset + 0.6 * norm_mfcc
            
            # --- 峰值排序 ---
            
            points = []
            for i, score in enumerate(combined_score):
                points.append((times[i], score))
            
            # 按综合分数排序
            sorted_points = sorted(points, key=lambda x: x[1], reverse=True)
            
            final_audio_cuts = []
            avg_score = np.mean(combined_score)
            std_score = np.std(combined_score)
            # 降低阈值，更灵敏地捕捉异常
            threshold_base = avg_score + 2.0 * std_score
            
            for t, score in sorted_points:
                if score < threshold_base: 
                    continue
                # 距离过滤 (1秒内只报最强点)
                is_near = False
                for existing_t, _ in final_audio_cuts:
                    if abs(t - existing_t) < 1.0:
                        is_near = True
                        break
                if not is_near:
                    final_audio_cuts.append((t, score))
                    if len(final_audio_cuts) >= 8: # 增加检测点数量，避免遗漏
                        break
            
            print(f"    [音频分析结果]")
            if not final_audio_cuts:
                print("    - 结果: 音频特征平稳。")
                return []
            else:
                print(f"    - 结果: 发现声学特征断层，按【显著性】排序：")
                for i, (t, score) in enumerate(final_audio_cuts):
                    print(f"      [{i+1}] 时间: {format_timestamp(t)} | 突变分: {score:.3f}")
            
            # 绘图
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            librosa.display.waveshow(y, sr=sr, alpha=0.6)
            plt.title('波形图 (Waveform) - 红色虚线为疑似点')
            for t, _ in final_audio_cuts:
                plt.axvline(x=t, color='r', linestyle='--', alpha=0.8)

            plt.subplot(2, 1, 2)
            plt.plot(times, combined_score, label='综合特征突变 (Score)', color='green')
            plt.title('声学特征变化率 (MFCC + Energy) - 峰值即为断层')
            plt.axhline(y=threshold_base, color='gray', linestyle=':', label='动态阈值')
            for t, _ in final_audio_cuts:
                plt.axvline(x=t, color='r', linestyle='--', alpha=0.5)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig("audio_check_smart.png")
            print(f"    [图片] 详细分析图已保存至: audio_check_smart.png")
            
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
        print("\n=== 音视频拼接取证工具 (MFCC声纹增强版) ===")
        print("请输入文件路径 (支持 MP4, AVI, MP3, WAV 等):")
        target_file = input(">>> ").strip().strip("'").strip('"')

    if target_file and os.path.exists(target_file):
        tool = MediaForensicsTool(target_file)
        
        # 1. 元数据及类型检测
        tool.analyze_metadata()
        
        # 2. 视频检测 (如果是视频)
        video_cuts = []
        if tool.has_video:
            video_cuts = tool.detect_video_cuts_smart()
            
            check_points = [1.0]
            if video_cuts:
                check_points = video_cuts[:2] 
                print(f"\n[i] 重点对前 {len(check_points)} 个可疑点进行 ELA 篡改验证...")
            
            for t in check_points:
                tool.perform_ela_on_frame(t)
        else:
            print("\n[i] 纯音频文件，跳过视频画面分析模块。")
        
        # 3. 音频检测
        audio_cuts = tool.analyze_audio_smart()
        
        # 4. 综合判定
        print("\n=== 🏁 最终取证结论 (基于 Top 排名) ===")
        
        if tool.has_video:
            primary_match = False
            if video_cuts and audio_cuts:
                v_top1 = video_cuts[0]
                for a_cut in audio_cuts[:3]: 
                    if abs(v_top1 - a_cut) < 1.0:
                        print(f"✅ 【确凿证据】 视频最强突变点与音频断层重合！")
                        print(f"   >>> 拼接点极大概率在: {format_timestamp(v_top1)} <<<")
                        primary_match = True
                        break
            
            if not primary_match:
                if video_cuts: print(f"⚠️ 【疑似拼接】 视频画面在 {format_timestamp(video_cuts[0])} 处有最大突变。")
                if audio_cuts: print(f"⚠️ 【疑似拼接】 音频波形在 {format_timestamp(audio_cuts[0])} 处有最大断层。")
        elif tool.has_audio:
            # 纯音频模式下的结论
            if audio_cuts:
                print(f"⚠️ 【疑似剪辑】 检测到音频声纹存在 {len(audio_cuts)} 处显著断层/突变。")
                print(f"   最显著的拼接点可能在: {format_timestamp(audio_cuts[0])}")
                print("   请参考生成的特征图 (audio_check_smart.png) 观察绿色曲线的尖峰。")
            else:
                print("✅ 【低风险】 音频波形与声纹连续性良好，未检测到明显的硬剪辑痕迹。")

    else:
        print("[!] 文件不存在")
