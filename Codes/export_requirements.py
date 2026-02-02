import subprocess
import sys


def export_env():
    try:
        result = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        packages = result.decode('utf-8')

        linux_friendly_packages = []
        for line in packages.split('\n'):
            line = line.strip()
            if not line: continue
            if any(x in line.lower() for x in ['pywin32', 'pypiwin32', 'win-inet', 'wincertstore']):
                continue

            linux_friendly_packages.append(line)

        with open('requirements.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(linux_friendly_packages))

    except subprocess.CalledProcessError as e:
        print(e)


if __name__ == "__main__":
    export_env()