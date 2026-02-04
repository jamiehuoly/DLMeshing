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

    def update_boundary_conditions(self, config):
        """
        Use foamDictionary to edit (0/U, 0/p)
        """
        boundaries = config.get("boundaries", {})
        if not boundaries:
            return

        print("🔧 Updating Boundary Conditions using 'foamDictionary'...")

        for patch_name, settings in boundaries.items():
            u_file = os.path.join(self.case_dir, "0", "U")
            if os.path.exists(u_file):
                # default value
                u_type = "fixedValue"
                default_uvalue = "uniform (0 0 0)"
                u_value = None

                if patch_name == "inlet":
                    u_type = settings.get("u_type", "fixedValue")
                    vals = settings.get("u_value", [0, 0, 0])
                    if isinstance(vals, list):
                        u_value = f"uniform ({vals[0]} {vals[1]} {vals[2]})"
                    else:
                        u_value = f"uniform {vals}"

                elif patch_name == "outlet":
                    u_type = settings.get("u_type", "zeroGradient")

                elif patch_name == "wall":
                    u_type = settings.get("u_type", "noSlip")

                # edit 'type' & 'value'
                self._set_foam_entry(u_file, f"boundaryField.{patch_name}.type", u_type)
                if u_value:
                    self._set_foam_entry(u_file, f"boundaryField.{patch_name}.value", u_value)
                else:
                    if self._test_foam_entry(u_file, f"boundaryField.{patch_name}.value"):
                        self._set_foam_entry(u_file, f"boundaryField.{patch_name}.value", default_uvalue)

            p_file = os.path.join(self.case_dir, "0", "p")
            if os.path.exists(p_file):
                # default value
                p_type = "zeroGradient"
                default_pvalue = "uniform 0"
                p_value = None

                if patch_name == "inlet":
                    p_type = settings.get("p_type", "zeroGradient")
                elif patch_name == "outlet":
                    p_type = settings.get("p_type", "fixedValue")
                    p_value = settings.get("p_value", default_pvalue)
                elif patch_name == "wall":
                    p_type = settings.get("p_type", "zeroGradient")

                self._set_foam_entry(p_file, f"boundaryField.{patch_name}.type", p_type)
                if p_value:
                    self._set_foam_entry(p_file, f"boundaryField.{patch_name}.value", p_value)
                else:
                    if self._test_foam_entry(p_file, f"boundaryField.{patch_name}.value"):
                        self._set_foam_entry(p_file, f"boundaryField.{patch_name}.value", default_pvalue)

    def _set_foam_entry(self, file_rel_path, entry, value):
        """
        Modify settings by using foamDictionary
        """
        # 构造命令: foamDictionary 0/U -entry boundaryField.inlet.type -set fixedValue
        cmd = f"foamDictionary {file_rel_path} -entry {entry} -set '{str(value)}'"
        self._run_cmd(cmd, log_name="foamDictionary")

    def _test_foam_entry(self, file_rel_path, entry):
        """
        Test value exist by using foamDictionary
        """
        cmd = f"foamDictionary {file_rel_path} -entry {entry}"

        result = subprocess.run(
                cmd,
                shell=True,
                cwd=self.case_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False
            )

        if result.returncode != 0:
            return False

        return True

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

    def create_dummy_foam(self):
        self._run_cmd(f"touch {self.case_dir}/case.foam")

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