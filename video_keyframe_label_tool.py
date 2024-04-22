# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/20 00:51
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import time
import json
import queue
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import os
import shutil
import threading


class VideoKeyFrameEventAnnotator(object):
    """
    视频关键帧事件标注工具
    """

    def __init__(self, main_window):
        # 初始化标签字典
        self.event_kerframe_class_to_index_dict = {
            "落地(下)": 0,
            "落地(上)": 1,
            "触网": 2,
            "出画面": 3,
            "正手击球(下)": 4,
            "正手击球(上)": 5,
            "反手击球(下)": 6,
            "反手击球(上)": 7,
            "发球(下)": 8,
            "发球(上)": 9
        }

        self.event_kerframe_index_to_class_dict = {
            0: "落地(下)",
            1: "落地(上)",
            2: "触网",
            3: "出画面",
            4: "正手击球(下)",
            5: "正手击球(上)",
            6: "反手击球(下)",
            7: "反手击球(上)",
            8: "发球(下)",
            9: "发球(上)"
        }

        # 初始化一重要参数
        self.cap = None
        self.video_path = None
        self.frame_count = -1
        self.video_result_save_root_dir = None
        self.video_result_save_dir = None
        self.video_frames_save_dir = None
        self.video_annotate_json_file_path = None
        self.annotate_dict = {}
        self.current_frame_id = -1
        self.play_interval = 1 / 30
        self.frame_width = 960
        self.frame_height = 540
        self.width_diff = 240
        self.height_diff = 24

        # 初始化自动播放的控制信号
        self.is_playing_evnet = threading.Event()
        self.is_playing_evnet.clear()
        # 初始化读取和处理后图像的线程安全的队列
        self.frame_imgtk_queue = queue.Queue(maxsize=1000)
        # 初始化读取和处理图像的线程
        self.auto_play_read_process_thread = threading.Thread(target=self._auto_play_read_process, args=(self.is_playing_evnet, self.frame_imgtk_queue), daemon=True)
        self.auto_play_render_thread = threading.Thread(target=self._auto_play_render, args=(self.is_playing_evnet, self.frame_imgtk_queue), daemon=True)
        self.auto_play_read_process_thread.start()
        self.auto_play_render_thread.start()

        # 初始化写标注文件的任务队列
        self.write_json_file_queue = queue.Queue(maxsize=10)
        # 初始化修改标注字典的线程锁
        self.annotate_dict_lock = threading.Lock()
        # 初始化写标注文件的线程
        self.write_json_file_thread = threading.Thread(target=self.write_json_file, args=(self.write_json_file_queue, self.annotate_dict_lock), daemon=True)
        self.write_json_file_thread.start()

        # 初始化主窗口
        self.main_window = main_window
        self.main_window.title("视频关键帧事件标注工具")

        # 视频帧图像显示区域
        self.display_label = tk.Label(self.main_window)
        self.display_label.pack(side="left", padx=10, pady=10, anchor="w")
        self.show_frame(frame_idx=None)  # 打开时显示空白图像

        # 右边控制区域
        self.right_control_frame = tk.Frame(self.main_window)
        self.right_control_frame.pack(side="left", after=self.display_label, padx=10, pady=10, fill="y", anchor="e")

        # 功能按钮区域
        self.button_frame = tk.Frame(self.right_control_frame)
        self.button_frame.pack(side="top", anchor="n")

        # 依次添加功能按钮
        # 打开视频按钮
        self.open_video_button = tk.Button(self.button_frame, text="打开视频", command=self.load_video, width=10).grid(row=0, column=0, padx=5, pady=5, sticky='nw')
        # 开始解析按钮
        self.start_extract_button = tk.Button(self.button_frame, text="解析视频", command=self.extract_video, width=10).grid(row=0, column=1, padx=5, pady=5, sticky='nw')
        # 上一帧按钮
        self.pre_frame_button = tk.Button(self.button_frame, text="上一帧", command=lambda: self.change_displayed_frame(-1), width=10).grid(row=1, column=0, padx=5, pady=5, sticky='nw')
        self.main_window.bind("<Left>", lambda event: self.change_displayed_frame(-1))
        # 下一帧按钮
        self.next_frame_button = tk.Button(self.button_frame, text="下一帧", command=lambda: self.change_displayed_frame(1), width=10).grid(row=1, column=1, padx=5, pady=5, sticky='nw')
        self.main_window.bind("<Right>", lambda event: self.change_displayed_frame(1))
        # 1倍速播放按钮
        self.x1_speed_play_button = tk.Button(self.button_frame, text="1倍速播放", command=lambda: self._x_speed_play(1), width=10).grid(row=2, column=0, padx=5, pady=5, sticky='nw')
        # 2倍速播放按钮
        self.x2_speed_play_button = tk.Button(self.button_frame, text="2倍速播放", command=lambda: self._x_speed_play(2), width=10).grid(row=2, column=1, padx=5, pady=5, sticky='nw')
        # 3倍速播放按钮
        self.x3_speed_play_button = tk.Button(self.button_frame, text="3倍速播放", command=lambda: self._x_speed_play(3), width=10).grid(row=3, column=0, padx=5, pady=5, sticky='nw')
        # 暂停/播放按钮
        self.toggle_pause_button = tk.Button(self.button_frame, text="暂停/播放", command=self.toggle_pause, width=10).grid(row=3, column=1, padx=5, pady=5, sticky='nw')
        self.main_window.bind("<Down>", lambda event: self.toggle_pause())

        # 创建一个水平分界线
        separator = tk.Frame(self.button_frame, height=2, bd=1, relief="sunken", bg="black")
        separator.grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=10)

        # 落地(下)标注按钮
        self.bounce_bottom_button = tk.Button(self.button_frame, text="落地(下)", command=lambda: self._annotate_label(self.event_kerframe_class_to_index_dict["落地(下)"]), width=10)
        self.bounce_bottom_button.grid(row=5, column=0, padx=5, pady=5, sticky='nw')
        # 落地(上)标注按钮
        self.bounce_top_button = tk.Button(self.button_frame, text="落地(上)", command=lambda: self._annotate_label(self.event_kerframe_class_to_index_dict["落地(上)"]), width=10)
        self.bounce_top_button.grid(row=5, column=1, padx=5, pady=5, sticky='nw')
        # 触网标注按钮
        self.catch_net_button = tk.Button(self.button_frame, text="触网", command=lambda: self._annotate_label(self.event_kerframe_class_to_index_dict["触网"]), width=10)
        self.catch_net_button.grid(row=6, column=0, padx=5, pady=5, sticky='nw')
        # 出画面标注按钮
        self.come_out_button = tk.Button(self.button_frame, text="出画面", command=lambda: self._annotate_label(self.event_kerframe_class_to_index_dict["出画面"]), width=10)
        self.come_out_button.grid(row=6, column=1, padx=5, pady=5, sticky='nw')
        # 正手击球(下)标注按钮
        self.forehand_bottom_button = tk.Button(self.button_frame, text="正手击球(下)", command=lambda: self._annotate_label(self.event_kerframe_class_to_index_dict["正手击球(下)"]), width=10)
        self.forehand_bottom_button.grid(row=7, column=0, padx=5, pady=5, sticky='nw')
        # 正手击球(上)标注按钮
        self.forehand_top_button = tk.Button(self.button_frame, text="正手击球(上)", command=lambda: self._annotate_label(self.event_kerframe_class_to_index_dict["正手击球(上)"]), width=10)
        self.forehand_top_button.grid(row=7, column=1, padx=5, pady=5, sticky='nw')
        # 反手击球(下)标注按钮
        self.backhand_bottom_button = tk.Button(self.button_frame, text="反手击球(下)", command=lambda: self._annotate_label(self.event_kerframe_class_to_index_dict["反手击球(下)"]), width=10)
        self.backhand_bottom_button.grid(row=8, column=0, padx=5, pady=5, sticky='nw')
        # 反手击球(上)标注按钮
        self.backhand_top_button = tk.Button(self.button_frame, text="反手击球(上)", command=lambda: self._annotate_label(self.event_kerframe_class_to_index_dict["反手击球(上)"]), width=10)
        self.backhand_top_button.grid(row=8, column=1, padx=5, pady=5, sticky='nw')
        # 发球(下)标注按钮
        self.serve_bottom_button = tk.Button(self.button_frame, text="发球(下)", command=lambda: self._annotate_label(self.event_kerframe_class_to_index_dict["发球(下)"]), width=10)
        self.serve_bottom_button.grid(row=9, column=0, padx=5, pady=5, sticky='nw')
        # 发球(上)标注按钮
        self.serve_top_button = tk.Button(self.button_frame, text="发球(上)", command=lambda: self._annotate_label(self.event_kerframe_class_to_index_dict["发球(上)"]), width=10)
        self.serve_top_button.grid(row=9, column=1, padx=5, pady=5, sticky='nw')

        # 显示当前标签的文本框
        self.display_current_label_entry = tk.Entry(self.button_frame, readonlybackground="lightblue", fg="red", font=('Times New Roman', 16), width=1, borderwidth=2, state="readonly")
        self.display_current_label_entry.grid(row=10, column=0, rowspan=2, columnspan=2, padx=5, pady=10, ipady=10, sticky='ew')

        # 清除标注按钮
        self.clear_label_button = tk.Button(self.button_frame, text="清除标注", command=self.clear_label, width=10)
        self.clear_label_button.grid(row=12, column=0, padx=5, pady=5, sticky='nw')

        # 底部帧数显示区域
        self.frame_count_frame = tk.Frame(self.right_control_frame)
        self.frame_count_frame.pack(side="bottom", anchor="w")

        # 为中间空白部分配置权重
        self.frame_count_frame.grid_rowconfigure(0, weight=1)

        # 添加跳转当前帧控件
        self.jump_frame_id_frame = tk.Frame(self.frame_count_frame)
        self.jump_frame_id_frame.grid(row=1, column=0, padx=5, pady=0, sticky='w')
        # 添加跳转输入框
        self.jump_frame_id_input_entry = tk.Entry(self.jump_frame_id_frame, readonlybackground="white", width=4, borderwidth=1)
        self.jump_frame_id_input_entry.grid(row=0, column=0, padx=1, pady=0, ipady=2, sticky='w')
        # 跳转按钮
        self.jump_current_frame_id_button = tk.Button(self.jump_frame_id_frame, text="Go", command=self.jump_current_frame_id, width=2, height=1)
        self.jump_current_frame_id_button.grid(row=0, column=1, padx=1, pady=0, sticky='w')

        # 添加帧数显示控件
        self.frame_id_text_frame = tk.Frame(self.frame_count_frame, width=10)
        self.frame_id_text_frame.grid(row=2, column=0, padx=0, pady=0, sticky='w')
        # 添加当前帧数文本
        self.current_frame_id_label = tk.Label(self.frame_id_text_frame, anchor="e")
        self.current_frame_id_label.grid(row=0, column=0, padx=0, pady=0, sticky='s')
        self.current_frame_id_label.config(text="")
        # 添加斜杠
        self.slash_label = tk.Label(self.frame_id_text_frame, anchor="c", width=1)
        self.slash_label.grid(row=0, column=1, padx=0, pady=0, sticky='s')
        self.slash_label.config(text="/")
        # 添加总帧数文本
        self.total_frame_label = tk.Label(self.frame_id_text_frame, anchor="w", width=10)
        self.total_frame_label.grid(row=0, column=2, padx=0, pady=0, sticky='s')
        self.total_frame_label.config(text="")

    def show_frame(self, frame_idx=None, imgtk=None):
        if imgtk is not None:
            self.display_label.config(image=imgtk)
            self.display_label.image = imgtk
        elif frame_idx is None:
            # 创建一个空白的透明图像作为label的大小占位符
            empty_image = Image.new('RGB', (self.frame_width, self.frame_height), (255, 255, 255))
            imgtk = ImageTk.PhotoImage(empty_image)
            self.display_label.config(image=imgtk)
            self.display_label.image = imgtk
        else:
            if self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)
                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.display_label.config(image=imgtk)
                    self.display_label.image = imgtk
        # 展示标注信息
        if frame_idx is not None:
            self.current_frame_id_label.config(text=str(frame_idx))
            self._display_label(frame_idx)

    def _create_progress_window_bar(self):
        # 创建顶层窗口
        progress_window = tk.Toplevel(self.main_window)
        progress_window.title("视频解析进度")
        progress_window.geometry("300x100+{}+{}".format(self.main_window.winfo_x() + self.main_window.winfo_width() // 2 - 150, self.main_window.winfo_y() + self.main_window.winfo_height() // 2 - 50))
        # 创建关闭进度条窗口的事件
        close_progress_window_event = threading.Event()
        # 绑定关闭进度条窗口事件
        progress_window.protocol("WM_DELETE_WINDOW", lambda: close_progress_window_event.set())
        # 创建进度条
        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=280, mode="determinate")
        progress_bar['maximum'] = 100  # 进度值最大值
        progress_bar['value'] = 0  # 进度值初始值
        progress_bar.pack(pady=20, padx=10)
        progress_bar.update_idletasks()

        return progress_bar, progress_window, close_progress_window_event

    def _extract_video(self, progress_bar, progress_window, close_progress_window_event):
        # 打开临时视频流
        temp_cap = cv2.VideoCapture(self.video_path)
        count = 0
        while True:
            ret, frame = temp_cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(self.video_frames_save_dir, f'frame_{count:08d}.jpg'), frame)
            count += 1
            if int(100 * count / self.frame_count) > int(100 * (count - 1) / self.frame_count):
                progress_bar['value'] = int(100 * count / self.frame_count)  # 更新进度条的当前进度
                progress_window.update_idletasks()  # 更新GUI
            # 判断是否结束线程
            if close_progress_window_event.is_set():
                # 关闭临时视频流
                temp_cap.release()
                # 销毁顶层窗口和进度条
                progress_window.destroy()
        # 关闭临时视频流
        temp_cap.release()
        # 销毁顶层窗口和进度条
        progress_window.destroy()

    def extract_video(self):
        self.is_playing_evnet.clear()
        # 判断是否存在当前目录且目录下的图像数等于视频帧数，满足则不用重新解析了
        if (self.video_frames_save_dir is None) or (os.path.exists(self.video_frames_save_dir) and len(os.listdir(self.video_frames_save_dir)) == self.frame_count):
            return
        # 如果没有解析过或者解析不完全，则执行视频解析
        if os.path.exists(self.video_frames_save_dir):
            shutil.rmtree(self.video_frames_save_dir)
        os.makedirs(self.video_frames_save_dir, exist_ok=False)  # 创建视频所有帧图像的存储目录
        # 创建顶层窗口和进度条
        progress_bar, progress_window, close_progress_window_event = self._create_progress_window_bar()
        # 启动一个线程解析视频
        extract_video_thread = threading.Thread(target=self._extract_video, args=(progress_bar, progress_window, close_progress_window_event), daemon=True)
        extract_video_thread.start()

    def update_main_window_position(self):
        self.main_window_width = self.frame_width + self.width_diff
        self.main_window_height = self.frame_height + self.height_diff
        self.main_window_offset_x = int((self.main_window.winfo_screenwidth() - self.main_window_width) / 2)
        self.main_window_offset_y = int((self.main_window.winfo_screenheight() - self.main_window_height) / 2)
        self.main_window.geometry(f"{self.main_window_width}x{self.main_window_height}+{self.main_window_offset_x}+{self.main_window_offset_y}")

    def load_video(self):
        self.is_playing_evnet.clear()
        self.video_path = filedialog.askopenfilename()
        if self.video_path:
            # 打开视频流
            self.cap = cv2.VideoCapture(self.video_path)
            # 默认显示第一帧
            self.current_frame_id = 0
            self.current_frame_id_label.config(text=str(self.current_frame_id))
            self.show_frame(frame_idx=self.current_frame_id)
            # 获取视频总帧数
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.total_frame_label.config(text=str(self.frame_count))
            # 选择解析和标注结果存放目录
            self.video_result_save_root_dir = filedialog.askdirectory()
            if self.video_result_save_root_dir:
                # 计算当前视频解析和标注结果存放目录
                self.video_result_save_dir = os.path.join(self.video_result_save_root_dir, os.path.basename(self.video_path).split('.')[0])
                os.makedirs(self.video_result_save_dir, exist_ok=True)  # 创建视频解析和标注结果存放目录
                # 计算当前视频解析存放目录
                self.video_frames_save_dir = os.path.join(self.video_result_save_dir, "total_frames")
                # 计算当前视频标注json文件路径
                self.video_annotate_json_file_path = os.path.join(self.video_result_save_dir, "keyframe_label.json")
                # 如果当前视频标注json文件路径存在文件，则直接加载
                self.annotate_dict_lock.acquire()
                if os.path.isfile(self.video_annotate_json_file_path):
                    self.annotate_dict = self._load_json_file(self.video_annotate_json_file_path)
                else:
                    self.annotate_dict = {}
                self.annotate_dict_lock.release()

    def change_displayed_frame(self, step):
        self.is_playing_evnet.clear()
        self._change_displayed_frame(step)

    def _change_displayed_frame(self, step):
        if self.cap:
            current_frame_id = self.current_frame_id + step
            current_frame_id = max(0, min(self.frame_count - 1, current_frame_id))
            self.show_frame(frame_idx=current_frame_id)
            self.current_frame_id = current_frame_id

    def _auto_play_read_process(self, is_playing_evnet, frame_imgtk_queue):
        first_start = True
        while True:
            # 如果事件没有被设置，清空队列
            if not is_playing_evnet.is_set():
                # 一次播放结束
                first_start = True
                # 事件等待，只有事件被设置时，才会继续执行
                is_playing_evnet.wait()
            # 事件被设置，开始播放
            if first_start:
                if self.cap:
                    current_frame_id = self.current_frame_id + 1
                    current_frame_id = max(0, min(self.frame_count - 1, current_frame_id))
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_id)
                else:
                    is_playing_evnet.clear()
                    continue
                first_start = False
            # 开始读取视频帧
            t1 = time.time()
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)  # 转换帧图像格式
                frame_imgtk_queue.put(imgtk, block=True)  # 使用线程安全的队列存储图像
            else:
                if self.current_frame_id >= self.frame_count - 1:
                    is_playing_evnet.clear()
            t2 = time.time()
            # 阻塞一段时间
            time.sleep(max(0, self.play_interval - (t2 - t1)) / 2)

    def _auto_play_render(self, is_playing_evnet, frame_imgtk_queue):
        while True:
            # 如果事件没有被设置，清空队列
            if not is_playing_evnet.is_set():
                try:
                    while True:
                        frame_imgtk_queue.get_nowait()  # 不等待，如果队列空了就会抛出异常
                except queue.Empty:
                    # 事件等待，只有事件被设置时，才会继续执行
                    is_playing_evnet.wait()
            if not self.cap:
                is_playing_evnet.clear()
                continue
            # 从队列中获取图像
            t1 = time.time()
            imgtk = frame_imgtk_queue.get(block=True)
            self.current_frame_id += 1
            self.show_frame(frame_idx=self.current_frame_id, imgtk=imgtk)
            t2 = time.time()
            # 阻塞一段时间
            time.sleep(max(0, self.play_interval - (t2 - t1)))

    def _x_speed_play(self, speed):
        self.play_interval = 1 / (30 * speed)
        self.is_playing_evnet.set()

    def toggle_pause(self):
        if self.is_playing_evnet.is_set():
            self.is_playing_evnet.clear()
        else:
            self.is_playing_evnet.set()

    @staticmethod
    def _write_json_file(video_annotate_json_file_path, annotate_dict):
        # 判断文件路径是否合法
        if video_annotate_json_file_path is None:
            return
        # 将字典写入到JSON文件
        with open(video_annotate_json_file_path, 'w') as f:
            json.dump(annotate_dict, f, indent=4)  # 使用indent参数使输出的JSON文件格式化，便于阅读

    def write_json_file(self, write_json_file_queue, annotate_dict_lock):
        while True:
            ret = write_json_file_queue.get(block=True)
            annotate_dict_lock.acquire()
            self._write_json_file(self.video_annotate_json_file_path, self.annotate_dict)
            annotate_dict_lock.release()

    @staticmethod
    def _load_json_file(video_annotate_json_file_path):
        if video_annotate_json_file_path is not None and os.path.isfile(video_annotate_json_file_path):
            # 从JSON文件读取数据
            with open(video_annotate_json_file_path, 'r') as f:
                return json.load(f)
        else:
            return None

    def _display_label(self, frame_idx):
        # 判断帧索引是否合法
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return
        # 修改标签显示文本框可编辑
        self.display_current_label_entry.config(state="normal")
        # 删除标签显示文本框原本内容
        self.display_current_label_entry.delete(0, tk.END)
        # 当前帧有标注才显示
        if frame_idx in self.annotate_dict:
            self.display_current_label_entry.insert(0, str(self.annotate_dict[frame_idx]) + ": " + self.event_kerframe_index_to_class_dict[self.annotate_dict[frame_idx]])
        # 修改标签显示文本框不可编辑
        self.display_current_label_entry.config(state="readonly")

    def _annotate_label(self, index):
        # 如果有自动播放，暂停
        self.is_playing_evnet.clear()
        # 如果当前标注和原本的一致，则什么都不用做
        if self.current_frame_id in self.annotate_dict and self.annotate_dict[self.current_frame_id] == index:
            return
        # 修改标注字典
        self.annotate_dict[self.current_frame_id] = index
        # 刷新显示标注的文本框
        self._display_label(self.current_frame_id)
        # 自动保存
        self.write_json_file_queue.put(True, block=True)

    def clear_label(self):
        # 如果有自动播放，暂停
        self.is_playing_evnet.clear()
        # 尝试在字典中删除当前帧的标注
        self.annotate_dict_lock.acquire()
        ret = self.annotate_dict.pop(self.current_frame_id, None)
        self.annotate_dict_lock.release()
        # 如果当前帧没有标注，啥也不用干
        if ret is None:
            return
        # 如果删除成功，刷新显示标注的文本框
        self._display_label(self.current_frame_id)
        # 自动保存
        self.write_json_file_queue.put(True, block=True)

    def jump_current_frame_id(self):
        # 如果有自动播放，暂停
        self.is_playing_evnet.clear()
        # 获取跳转帧数数值
        jump_frame_id = self.jump_frame_id_input_entry.get()
        if jump_frame_id == "":
            return
        jump_frame_id = int(jump_frame_id)
        # 如果跳转帧数合法
        if 0 <= jump_frame_id < self.frame_count and jump_frame_id != self.current_frame_id:
            self.current_frame_id = jump_frame_id
            self.show_frame(frame_idx=self.current_frame_id)


if __name__ == '__main__':
    # 创建主窗口
    main_window = tk.Tk()
    app = VideoKeyFrameEventAnnotator(main_window)
    # 更新主窗口位置
    app.update_main_window_position()
    main_window.mainloop()
