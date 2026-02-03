import os
import subprocess
import shutil
import json
import re

# 1. 定义 Case 路径
# TEST SCRIPT
# CASE_PATH = "/PhD/DLMeshing"
CASE_PATH = "/home/zhuo/phd/test"
TEMPLATE_PATH = "../FoamTemplate"
MSH_FILE = "elbow.msh"
CONFIG_FILE = "case_config.json"


class OpenFoamAutomator:
    def __init__(self, case_dir):
        """
        Initialize
        :param case_dir: root directory of OpenFOAM case (should consist of 0, constant, system)
        """
        self.case_dir = os.path.abspath(case_dir)
        self.log_dir = os.path.join(self.case_dir, "logs")

        if not os.path.exists(self.case_dir):
            raise FileNotFoundError(f"Case directory not found: {self.case_dir}")

        os.makedirs(self.log_dir, exist_ok=True)
        print(f"🚀 OpenFOAM Automation initialized at: {self.case_dir}")

    def _run_cmd(self, cmd_str, log_name=None):
        """
        Execute Shell commands
        """
        print(f"   Wait... Executing: {cmd_str}")
        try:
            # 如果指定了 log_name，则把输出重定向到文件
            stdout_dest = subprocess.PIPE
            if log_name:
                log_file = os.path.join(self.log_dir, f"{log_name}.log")
                f_log = open(log_file, "w")
                stdout_dest = f_log

            # 核心：cwd=self.case_dir 确保命令在 case 目录下执行
            process = subprocess.run(
                cmd_str,
                shell=True,
                cwd=self.case_dir,
                stdout=stdout_dest,
                stderr=subprocess.PIPE,
                text=True,
                check=True  # 如果命令返回非0，抛出异常
            )

            if log_name:
                f_log.close()
                print(f"   ✅ Done. Log saved to: logs/{log_name}.log")
            else:
                print("   ✅ Done.")

        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error executing command: {cmd_str}")
            print(f"   Error Details: {e.stderr}")
            raise e

    def prepare_compulsory_folders(self):
        self.prepare_target_folder("0")
        self.prepare_target_folder("constant")
        self.prepare_target_folder("system")
        print("Finished preparing compulsory folders.")

    def prepare_target_folder(self, folder_name: str):
        """
        Initialize directories
        """
        folder_path = os.path.join(self.case_dir, folder_name)
        folder_template = os.path.join(TEMPLATE_PATH, folder_name)

        if not os.path.exists(folder_path):
            shutil.copytree(folder_template, folder_path)
        else:
            print(f"Target folder: {folder_name} already exists. Skipping.")

    def import_gmsh(self, msh_file):
        """
        Execute gmshToFoam
        """
        # 确保 msh 文件路径是绝对路径，或者是相对于 case 的
        if not os.path.isabs(msh_file):
            msh_file = os.path.join(self.case_dir, msh_file)

        cmd = f"gmshToFoam {msh_file}"
        self._run_cmd(cmd, log_name="gmshToFoam")

    def scale_mesh(self, scale_factor=0.001):
        """
        Execute transformPoints (mm -> m)
        """
        scale_vec = f"({scale_factor} {scale_factor} {scale_factor})"
        cmd = f"transformPoints 'scale={scale_vec}'"
        self._run_cmd(cmd, log_name="transformPoints")

    def check_mesh(self):
        """
        Execute checkMesh
        """
        self._run_cmd("checkMesh", log_name="checkMesh")

    # ToDo: Not sufficiently tested, may occur problems when testing complex geometries
    def update_boundary_conditions(self, config_file):
        """
        Modify U and p based on config file
        """
        if not os.path.exists(config_file):
            print(f"⚠️ Config file {config_file} not found. Skipping BC update.")
            return

        with open(config_file, 'r') as f:
            config = json.load(f)

        boundaries = config.get("boundaries", {})
        if not boundaries:
            print("Boundaries in config file is empty!")

        for field_name in ["U", "p"]:
            file_path = os.path.join(self.case_dir, "0", field_name)
            if not os.path.exists(file_path):
                continue

            print(f"🔧 Updating {field_name} boundary conditions...")

            with open(file_path, 'r') as f:
                content = f.read()

            # 遍历 JSON 中的每个边界 (inlet, outlet...)
            for patch_name, settings in boundaries.items():
                # 判断当前处理的是 U 还是 p 的配置
                # 假设 JSON 结构里可能有区分，或者我们根据 field_name 猜测
                # 这里做一个简化逻辑：
                # 如果是 U 文件，且 JSON 里定义了 velocity (或者 type 是 fixedValue)
                # 这部分逻辑需要根据你的 case_config.json 结构定制

                # 构造 OpenFOAM 的 Block 正则表达式
                # 寻找 boundaryField { ... inlet { ... } ... }
                # 这是一个简化的文本替换逻辑

                # 1. 构造新的 patch 内容
                new_patch_block = f"\n    {patch_name}\n    {{\n"

                # 针对 U 场
                if field_name == "U":
                    if patch_name == "inlet":
                        # 特殊处理 inlet
                        u_type = settings.get("type", "fixedValue")
                        # 假设 config 里的 value 是 list [x, y, z]
                        if "value" in settings and isinstance(settings["value"], list):
                            vx, vy, vz = settings["value"]
                            u_val = f"uniform ({vx} {vy} {vz})"
                        else:
                            u_val = "uniform (0 0 0)"

                        new_patch_block += f"        type            {u_type};\n"
                        new_patch_block += f"        value           {u_val};\n"

                    elif patch_name == "outlet":
                        new_patch_block += "        type            zeroGradient;\n"

                    elif patch_name == "wall":
                        new_patch_block += "        type            noSlip;\n"

                # 针对 p 场
                elif field_name == "p":
                    if patch_name == "inlet":
                        new_patch_block += "        type            zeroGradient;\n"
                    elif patch_name == "outlet":
                        new_patch_block += "        type            fixedValue;\n"
                        new_patch_block += "        value           uniform 0;\n"
                    elif patch_name == "wall":
                        new_patch_block += "        type            zeroGradient;\n"

                new_patch_block += "    }"

                # 2. 使用正则表达式替换原有 block
                # 匹配: patch_name \n { ... }
                # 注意：OpenFOAM 的花括号嵌套很难完美匹配，这里假设标准格式
                pattern = r'(\s+' + re.escape(patch_name) + r'\s*\{)[^}]*(\})'

                # 如果找不到这个 patch (比如 gmsh 里的名字和 config 不一致)，跳过
                if re.search(pattern, content, re.DOTALL):
                    # 只替换大括号里的内容，或者整个块
                    # 这里为了简单，我们用正则找到这个块，然后替换它
                    # 这是一个粗暴但有效的方法：直接替换整个 patch 定义
                    content = re.sub(pattern, new_patch_block, content, count=1, flags=re.DOTALL)
                    print(f"   - Updated patch: {patch_name}")
                else:
                    print(f"   ⚠️ Patch '{patch_name}' not found in 0/{field_name}")

            with open(file_path, 'w') as f:
                f.write(content)

    def run_solver(self, solver_name="foamRun"):
        """
        Run solver
        """
        print(f"🔥 Running Solver: {solver_name} ...")
        self._run_cmd(solver_name, log_name=solver_name)

    def export_vtk(self):
        """
        Export VTK files
        """
        print("💾 Exporting to VTK...")
        self._run_cmd("foamToVTK", log_name="foamToVTK")

        vtk_dir = os.path.join(self.case_dir, "VTK")
        if os.path.exists(vtk_dir):
            print(f"   ✅ VTK files are located in: {vtk_dir}")
        else:
            print("   ⚠️ VTK directory was not created. Check logs.")

    def clean_redundants(self):
        self._run_cmd("foamListTimes -rm")
        print("Done cleaning foam redundants.")

    def clean_logs(self):
        log_path = os.path.join(self.case_dir, "logs")
        self._run_cmd(f"zip -r {log_path}/log.zip {log_path}")
        self._run_cmd(f"rm -rf {log_path}/*.log")
        print("Done cleaning logs.")

# if __name__ == "__main__":
#
#     runner = OpenFoamAutomator(CASE_PATH)
#
#     try:
#         runner.clean_logs()
#         runner.prepare_compulsory_folders()
#
#         runner.import_gmsh(MSH_FILE)
#         runner.scale_mesh(0.001)  # mm to m
#         runner.check_mesh()
#
#         runner.update_boundary_conditions(CONFIG_FILE)
#
#         runner.run_solver("foamRun")
#
#         runner.export_vtk()
#
#         print("\n🎉 All OpenFOAM tasks completed successfully!")
#
#     except Exception as e:
#         print(f"\n❌ Pipeline failed: {e}")