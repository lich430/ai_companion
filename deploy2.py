#!/usr/bin/env python3
"""Fix wheel and install uiautomator2"""
import paramiko

HOST = "192.168.250.240"
USER = "root"
PASSWORD = "admin"

def run_ssh(cmd):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=30)
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=300)
    out = stdout.read().decode()
    err = stderr.read().decode()
    ssh.close()
    return out, err

# 1. Reinstall setuptools and wheel
print("Reinstalling setuptools and wheel...")
out, err = run_ssh("pip3 install setuptools wheel --upgrade --force-reinstall --break-system-packages 2>&1")
print(out[-500:])

# 2. Install adbutils
print("\nInstalling adbutils...")
out, err = run_ssh("pip3 install adbutils --break-system-packages 2>&1")
print(out[-800:])

# 3. Install uiautomator2
print("\nInstalling uiautomator2...")
out, err = run_ssh("pip3 install uiautomator2 --break-system-packages 2>&1")
print(out[-800:])

# 4. Restart service
print("\nRestarting service...")
out, err = run_ssh("systemctl restart wechat-auto && sleep 2 && systemctl status wechat-auto --no-pager")
print(out[-1000:])

print("\nDone!")
