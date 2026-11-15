"""Windows controller for the MuMu reverse-bounty S flow."""
import queue
import json
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from ninja3_assistant.activities.bounty.flow import Flow
from ninja3_assistant.device.mumu import MuMu
from ninja3_assistant.runtime.runner import RunControl, run_flow
from ninja3_assistant.vision.vision import Vision

class Application:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('逆袭 S 助手')
        self.root.geometry('600x320')
        self.root.resizable(False, False)
        self.events = queue.Queue()
        self.stop = threading.Event()
        self.thread = None
        self.folder = tk.StringVar(value=r'D:\Program Files\Netease\MuMu')
        self.index = tk.StringVar(value='1')
        self.status = tk.StringVar(value='进入游戏的逆袭悬赏列表后启动。')
        self.backend = tk.StringVar(value='识别设备：等待启动')
        self.metrics = tk.StringVar(value='')
        self.kills = tk.StringVar(value='本次成功击杀：0 次')
        self.kill_count = 0

        main = ttk.Frame(self.root, padding=18)
        main.pack(fill='both', expand=True)
        ttk.Label(main, text='逆袭 S 自动加入', font=('Microsoft YaHei UI', 16, 'bold')).pack(anchor='w')
        ttk.Label(main, text='刷新 → 抢 S 队伍 → 准备 → 结算 → 继续').pack(anchor='w', pady=(4, 12))
        row = ttk.Frame(main); row.pack(fill='x')
        ttk.Label(row, text='MuMu 目录').pack(side='left')
        ttk.Entry(row, textvariable=self.folder).pack(side='left', fill='x', expand=True, padx=8)
        ttk.Button(row, text='选择', command=self.choose).pack(side='left')
        row = ttk.Frame(main); row.pack(fill='x', pady=10)
        ttk.Label(row, text='多开器实例编号').pack(side='left')
        ttk.Entry(row, textvariable=self.index, width=6).pack(side='left', padx=8)
        self.start_button = ttk.Button(row, text='启动流程', command=self.start)
        self.start_button.pack(side='left', padx=(16, 8))
        ttk.Button(row, text='停止', command=self.stop.set).pack(side='left')
        ttk.Label(main, textvariable=self.backend).pack(anchor='w', pady=(6, 2))
        ttk.Label(main, textvariable=self.status, wraplength=550).pack(anchor='w', pady=4)
        ttk.Label(main, textvariable=self.kills).pack(anchor='w')
        ttk.Label(main, textvariable=self.metrics).pack(anchor='w')
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.after(100, self.poll)

    def choose(self):
        path = filedialog.askdirectory(initialdir=self.folder.get())
        if path:
            self.folder.set(path)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        try:
            index = int(self.index.get())
            if index < 0: raise ValueError()
        except ValueError:
            self.status.set('请输入有效的 MuMu 实例编号。')
            return
        self.stop.clear()
        self.start_button.configure(state='disabled')
        self.status.set('正在连接模拟器并初始化识别…')
        self.thread = threading.Thread(target=self.run, args=(self.folder.get(), index), daemon=True)
        self.thread.start()

    def run(self, folder, index):
        def emit(kind, value):
            if kind == 'backend':
                value = '识别设备：' + value
            elif kind == 'metrics':
                value = f"识别 {value['fps']:.1f} 次/秒 · 已点击 {value['clicks']} 次"
            self.events.put((kind, value))

        run_flow(folder, index, RunControl(self.stop), emit)

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'backend': self.backend.set(value)
                elif kind == 'status': self.status.set(value)
                elif kind == 'metrics': self.metrics.set(value)
                elif kind == 'kill':
                    self.kill_count += value
                    self.kills.set(f'本次成功击杀：{self.kill_count} 次')
                elif kind == 'finished': self.start_button.configure(state='normal')
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()

if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--self-test':
        device = None
        try:
            import cv2
            import numpy as np
            from ninja3_assistant.vision.vision import ASSETS, Screen
            vision = Vision()
            device = MuMu(r'D:\Program Files\Netease\MuMu', 1)
            frame = device.capture()
            screen = vision.detect(frame)
            badge = cv2.imread(str(ASSETS / 'rank-s.png'), cv2.IMREAD_GRAYSCALE)
            sample = np.zeros((75, 110), dtype=np.uint8)
            sample[3:62, 15:89] = badge
            if not vision.rank_s(sample, 0, 0):
                raise RuntimeError('GPU/CPU 模型未识别 S 模板')
            test_flow = Flow()
            first = test_flow.step(Screen(board=True, a='off', s='on', ss='off', join=(389, 306)), 0)
            test_flow.completed(first, .1)
            shifted = test_flow.step(Screen(board=True, a='off', s='on', ss='off',
                                            join=(400, 306), card_cells=((95, 198),)), .2)
            if shifted.action != 'join' or shifted.point != (400, 306):
                raise RuntimeError('移动中的队伍未能继续点击')
            test_flow.completed(shifted, .3)
            result = test_flow.step(Screen(result=(650, 678)), .4)
            if result.action != 'confirm':
                raise RuntimeError('未识别结算操作')
            test_flow.completed(result, .5)
            test_flow.step(Screen(board=True, a='off', s='on', ss='off'), .6)
            if test_flow.successful_kills != 1:
                raise RuntimeError('击杀计数未更新')
            report = {'ok': True, 'backend': vision.backend, 'providers': vision.session.get_providers(),
                      'size': [frame.shape[1], frame.shape[0]], 'board': screen.board, 'rank_model': 'passed',
                      'rapid_retry': 'passed', 'settlement_counter': 'passed'}
        except Exception as error:
            report = {'ok': False, 'error': str(error)}
        finally:
            if device: device.close()
        Path(sys.argv[2]).write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')
    else:
        Application().mainloop()
