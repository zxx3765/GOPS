#  Copyright (c). All Rights Reserved.
#  GOPS ONNX Export Tool - GUI版本
#  
#  描述: 带图形界面的ONNX导出工具，使用文件对话框选择路径
#  创建日期: 2024年

"""
GUI版本的GOPS策略ONNX导出工具
使用tkinter创建简单的图形界面，通过文件对话框选择路径和参数
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from py2onnx import Py2ONNXRunner

class ONNXExportGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GOPS 策略 ONNX 导出工具")
        self.root.geometry("600x600")
        self.root.resizable(True, True)
        
        # 变量
        self.policy_dir = tk.StringVar()
        self.iteration = tk.StringVar(value="6078_opt")
        self.controller_name = tk.StringVar(value="NN_controller")
        self.save_path = tk.StringVar()
        self.export_format = tk.StringVar(value="onnx")
        self.opset_version = tk.StringVar(value="11")
        
        self.setup_ui()
        
    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标题
        title_label = ttk.Label(main_frame, text="GOPS 策略 ONNX 导出工具", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 策略目录选择
        ttk.Label(main_frame, text="训练策略目录:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.policy_dir, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="浏览...", command=self.browse_policy_dir).grid(row=1, column=2, pady=5)
        
        # 策略目录说明
        help_label = ttk.Label(main_frame, text="注：只需要选择到时间一级目录即可 (如: DDPG_250825-150421)", 
                              font=("Arial", 8), foreground="gray")
        help_label.grid(row=2, column=1, sticky=tk.W, padx=5, pady=(0, 10))
        
        # 迭代步数
        ttk.Label(main_frame, text="迭代步数:").grid(row=3, column=0, sticky=tk.W, pady=5)
        iteration_entry = ttk.Entry(main_frame, textvariable=self.iteration, width=20)
        iteration_entry.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 控制器名称
        ttk.Label(main_frame, text="控制器名称:").grid(row=4, column=0, sticky=tk.W, pady=5)
        controller_entry = ttk.Entry(main_frame, textvariable=self.controller_name, width=30)
        controller_entry.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 保存路径选择
        ttk.Label(main_frame, text="保存路径:").grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.save_path, width=50).grid(row=5, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="浏览...", command=self.browse_save_path).grid(row=5, column=2, pady=5)
        
        # 导出格式
        ttk.Label(main_frame, text="导出格式:").grid(row=6, column=0, sticky=tk.W, pady=5)
        format_combo = ttk.Combobox(main_frame, textvariable=self.export_format, 
                                   values=["onnx", "torchscript"], state="readonly", width=15)
        format_combo.grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)
        
        # OPSET版本
        ttk.Label(main_frame, text="OPSET版本:").grid(row=7, column=0, sticky=tk.W, pady=5)
        opset_entry = ttk.Entry(main_frame, textvariable=self.opset_version, width=10)
        opset_entry.grid(row=7, column=1, sticky=tk.W, padx=5, pady=5)
        
        # 分隔线
        ttk.Separator(main_frame, orient='horizontal').grid(row=8, column=0, columnspan=3, 
                                                           sticky=(tk.W, tk.E), pady=20)
        
        # 快速设置按钮
        quick_frame = ttk.LabelFrame(main_frame, text="快速设置", padding="10")
        quick_frame.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(quick_frame, text="设置为DDPG默认", 
                  command=self.set_ddpg_default).grid(row=0, column=0, padx=5)
        ttk.Button(quick_frame, text="设置为PPO默认", 
                  command=self.set_ppo_default).grid(row=0, column=1, padx=5)
        ttk.Button(quick_frame, text="清空所有", 
                  command=self.clear_all).grid(row=0, column=2, padx=5)
        
        # 导出按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=10, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="开始导出", command=self.export_model, 
                  style="Accent.TButton").pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="退出", command=self.root.quit).pack(side=tk.LEFT, padx=10)
        
        # 状态栏
        self.status_var = tk.StringVar(value="准备就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=11, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 配置权重
        main_frame.columnconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def browse_policy_dir(self):
        """浏览策略目录"""
        directory = filedialog.askdirectory(
            title="选择训练策略目录",
            initialdir=r"D:\Project\GOPS\results" if os.path.exists(r"D:\Project\GOPS\results") else "/"
        )
        if directory:
            self.policy_dir.set(directory)
            # 自动从路径中提取算法名称
            dirname = os.path.basename(directory)
            if "DDPG" in dirname.upper():
                self.controller_name.set("NN_DDPG_controller")
            elif "PPO" in dirname.upper():
                self.controller_name.set("NN_PPO_controller")
            elif "SAC" in dirname.upper():
                self.controller_name.set("NN_SAC_controller")
            
    def browse_save_path(self):
        """浏览保存路径"""
        directory = filedialog.askdirectory(
            title="选择保存目录",
            initialdir=r"D:\Project\SynologyDrive\imp_fcn_cal\RL_model" if os.path.exists(r"D:\Project\SynologyDrive\imp_fcn_cal\RL_model") else "/"
        )
        if directory:
            self.save_path.set(directory)
            
    def set_ddpg_default(self):
        """设置DDPG默认值"""
        self.policy_dir.set(r"D:\Project\GOPS\results\simu_quarter_sus_win\DDPG_250825-150421")
        self.iteration.set("6078_opt")
        self.controller_name.set("NN_DDPG_6078opt")
        self.save_path.set(r"D:\Project\SynologyDrive\imp_fcn_cal\RL_model")
        self.export_format.set("onnx")
        self.status_var.set("已设置DDPG默认参数")
        
    def set_ppo_default(self):
        """设置PPO默认值"""
        self.policy_dir.set(r"D:\Project\GOPS\results\simu_quarter_sus_win\DDPG_250825-220417")
        self.iteration.set("28000")
        self.controller_name.set("NN_PPO_28000")
        self.save_path.set(r"D:\Project\SynologyDrive\imp_fcn_cal\RL_model")
        self.export_format.set("onnx")
        self.status_var.set("已设置PPO默认参数")
        
    def clear_all(self):
        """清空所有输入"""
        self.policy_dir.set("")
        self.iteration.set("")
        self.controller_name.set("")
        self.save_path.set("")
        self.export_format.set("onnx")
        self.opset_version.set("11")
        self.status_var.set("已清空所有参数")
        
    def validate_inputs(self):
        """验证输入参数"""
        if not self.policy_dir.get():
            messagebox.showerror("错误", "请选择训练策略目录")
            return False
            
        if not os.path.exists(self.policy_dir.get()):
            messagebox.showerror("错误", "策略目录不存在")
            return False
            
        if not self.iteration.get():
            messagebox.showerror("错误", "请输入迭代步数")
            return False
            
        if not self.controller_name.get():
            messagebox.showerror("错误", "请输入控制器名称")
            return False
            
        if not self.save_path.get():
            messagebox.showerror("错误", "请选择保存路径")
            return False
            
        try:
            int(self.opset_version.get())
        except ValueError:
            messagebox.showerror("错误", "OPSET版本必须是数字")
            return False
            
        return True
        
    def export_model(self):
        """导出模型"""
        if not self.validate_inputs():
            return
            
        try:
            self.status_var.set("正在导出模型...")
            self.root.update()
            
            # 创建导出器
            runner = Py2ONNXRunner(
                log_policy_dir_list=[self.policy_dir.get()],
                trained_policy_iteration_list=[self.iteration.get()],
                export_controller_name=[self.controller_name.get()],
                save_path=[self.save_path.get()],
                export_format=self.export_format.get(),
                opset_version=int(self.opset_version.get()),
            )
            
            # 执行导出
            runner.export_policies()
            
            # 生成文件路径
            file_extension = ".onnx" if self.export_format.get() == "onnx" else ".pt"
            output_file = os.path.join(self.save_path.get(), 
                                     f"{self.controller_name.get()}{file_extension}")
            
            self.status_var.set(f"导出成功！")
            
            # 显示成功消息
            result = messagebox.askyesno(
                "导出成功", 
                f"模型已成功导出为:\n{output_file}\n\n是否要打开保存目录？"
            )
            
            if result:
                os.startfile(self.save_path.get())
                
        except Exception as e:
            self.status_var.set("导出失败")
            messagebox.showerror("导出失败", f"导出过程中发生错误:\n{str(e)}")
            
    def run(self):
        """运行GUI"""
        self.root.mainloop()

def main():
    """主函数"""
    app = ONNXExportGUI()
    app.run()

if __name__ == "__main__":
    main()